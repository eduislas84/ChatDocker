import os
import logging
import asyncio
import time
import hashlib
import psutil
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from groq import Groq
import aiofiles
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración segura
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
AUTHORIZED_USERS = set(map(int, os.getenv('AUTHORIZED_USERS', '').split(','))) if os.getenv('AUTHORIZED_USERS') else set()

# Validaciones de configuración
if not TELEGRAM_TOKEN or not GROQ_API_KEY:
    raise ValueError("Variables de entorno TELEGRAM_TOKEN y GROQ_API_KEY son requeridas")

# Configuración de límites
MAX_FILE_SIZE_MB = 50
MAX_ROWS = 100000
MAX_MEMORY_MB = 500
DATA_TTL_SECONDS = 3600  # 1 hora
MAX_CONCURRENT_USERS = 50
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW = 60  # segundos

# Configurar logging (SIN EMOJIS para evitar problemas de encoding)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def escape_markdown_v2(text: str) -> str:
    """Escapa caracteres especiales para MarkdownV2"""
    escape_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in escape_chars:
        text = text.replace(char, f'\\{char}')
    return text

class RateLimiter:
    """Control de límites de uso por usuario"""
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.user_requests = {}
    
    def is_allowed(self, user_id: int) -> bool:
        now = time.time()
        if user_id not in self.user_requests:
            self.user_requests[user_id] = []
        
        # Limpiar requests antiguos
        self.user_requests[user_id] = [
            req_time for req_time in self.user_requests[user_id]
            if now - req_time < self.window_seconds
        ]
        
        # Verificar límite
        if len(self.user_requests[user_id]) >= self.max_requests:
            return False
        
        # Registrar nueva request
        self.user_requests[user_id].append(now)
        return True

class QueryCache:
    """Cache inteligente para consultas similares"""
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
    
    def _get_cache_key(self, df_shape: tuple, query: str) -> str:
        df_hash = hashlib.md5(f"{df_shape}".encode()).hexdigest()[:8]
        query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()[:8]
        return f"{df_hash}_{query_hash}"
    
    def get(self, df_shape: tuple, query: str) -> Optional[str]:
        key = self._get_cache_key(df_shape, query)
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, df_shape: tuple, query: str, result: str):
        key = self._get_cache_key(df_shape, query)
        
        # Limpiar cache si está lleno
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = result
        self.access_times[key] = time.time()

class SafeOperationExecutor:
    """Ejecutor seguro de operaciones en DataFrames"""
    
    SAFE_OPERATIONS = {
        'mean': lambda df, col: df[col].mean() if col in df.columns else None,
        'sum': lambda df, col: df[col].sum() if col in df.columns else None,
        'max': lambda df, col: df[col].max() if col in df.columns else None,
        'min': lambda df, col: df[col].min() if col in df.columns else None,
        'count': lambda df, col: df[col].count() if col in df.columns else None,
        'nunique': lambda df, col: df[col].nunique() if col in df.columns else None,
        'shape': lambda df, col: df.shape,
        'info': lambda df, col: f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}",
        'head': lambda df, col: df.head().to_string(),
        'describe': lambda df, col: df.describe().to_string() if col is None else df[col].describe().to_string()
    }
    
    @classmethod
    def execute_safe(cls, df: pd.DataFrame, operation: str, column: str = None) -> Any:
        """Ejecuta operación segura en el DataFrame"""
        if operation not in cls.SAFE_OPERATIONS:
            raise ValueError(f"Operación '{operation}' no permitida")
        
        try:
            return cls.SAFE_OPERATIONS[operation](df, column)
        except Exception as e:
            raise ValueError(f"Error al ejecutar operación: {str(e)}")

class DataFrameManager:
    """Gestor optimizado de DataFrames con límites de memoria"""
    
    def __init__(self):
        self.dataframes = {}
        self.timestamps = {}
        self.file_paths = {}
    
    def add_dataframe(self, user_id: int, df: pd.DataFrame, file_path: str):
        """Agrega DataFrame con gestión de memoria"""
        self._cleanup_expired()
        self._enforce_memory_limits()
        
        self.dataframes[user_id] = df
        self.timestamps[user_id] = time.time()
        self.file_paths[user_id] = file_path
        
        logger.info(f"DataFrame agregado para usuario {user_id}. Memoria actual: {self._get_memory_usage():.1f}MB")
    
    def get_dataframe(self, user_id: int) -> Optional[pd.DataFrame]:
        """Obtiene DataFrame del usuario"""
        if user_id in self.dataframes:
            self.timestamps[user_id] = time.time()  # Actualizar acceso
            return self.dataframes[user_id]
        return None
    
    def remove_user(self, user_id: int):
        """Elimina datos del usuario"""
        if user_id in self.dataframes:
            del self.dataframes[user_id]
            del self.timestamps[user_id]
            
            # Limpiar archivo temporal
            if user_id in self.file_paths:
                try:
                    if os.path.exists(self.file_paths[user_id]):
                        os.remove(self.file_paths[user_id])
                except Exception as e:
                    logger.warning(f"Error al eliminar archivo temporal: {e}")
                del self.file_paths[user_id]
    
    def _cleanup_expired(self):
        """Limpia DataFrames expirados"""
        current_time = time.time()
        expired_users = [
            user_id for user_id, timestamp in self.timestamps.items()
            if current_time - timestamp > DATA_TTL_SECONDS
        ]
        
        for user_id in expired_users:
            logger.info(f"Limpiando datos expirados del usuario {user_id}")
            self.remove_user(user_id)
    
    def _get_memory_usage(self) -> float:
        """Calcula uso de memoria en MB"""
        total_memory = 0
        for df in self.dataframes.values():
            total_memory += df.memory_usage(deep=True).sum()
        return total_memory / (1024**2)
    
    def _enforce_memory_limits(self):
        """Aplica límites de memoria eliminando usuarios menos activos"""
        while self._get_memory_usage() > MAX_MEMORY_MB and self.dataframes:
            # Eliminar usuario con acceso más antiguo
            oldest_user = min(self.timestamps.keys(), key=self.timestamps.get)
            logger.warning(f"Límite de memoria excedido. Eliminando usuario {oldest_user}")
            self.remove_user(oldest_user)

class SecureTelegramBot:
    def __init__(self):
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        self.data_manager = DataFrameManager()
        self.query_cache = QueryCache()
        self.rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.operation_executor = SafeOperationExecutor()
        self.last_suggested_code = {}
    
    async def _check_authorization(self, update: Update) -> bool:
        """Verifica autorización del usuario"""
        if AUTHORIZED_USERS and update.effective_user.id not in AUTHORIZED_USERS:
            await update.message.reply_text("❌ No tienes autorización para usar este bot.")
            return False
        return True
    
    async def _check_rate_limit(self, update: Update) -> bool:
        """Verifica límites de uso"""
        if not self.rate_limiter.is_allowed(update.effective_user.id):
            await update.message.reply_text(
                f"⏱️ Has excedido el límite de {RATE_LIMIT_REQUESTS} consultas por minuto. "
                "Espera un momento antes de continuar."
            )
            return False
        return True
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /start"""
        if not await self._check_authorization(update):
            return
        
        message = (
            "🤖 *Bot CSV Seguro v2\\.0*\n\n"
            "📁 Envíame un archivo CSV \\(máx\\. 50MB, 100k filas\\)\n"
            "❓ Haz preguntas sobre tus datos\n"
            "🔒 Operaciones seguras y optimizadas\n\n"
            "*Comandos disponibles:*\n"
            "• /ayuda \\- Ver ejemplos de preguntas\n"
            "• /info \\- Información del archivo actual\n"
            "• /limpiar \\- Eliminar tus datos\n"
            "• /estado \\- Estado del sistema"
        )
        
        await update.message.reply_text(message, parse_mode="MarkdownV2")
    
    async def ayuda(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /ayuda"""
        if not await self._check_authorization(update):
            return
        
        user_id = update.effective_user.id
        df = self.data_manager.get_dataframe(user_id)
        
        if df is None:
            message = (
                "📚 Ejemplos de preguntas generales:\n\n"
                "• ¿Cuál es el promedio de la columna ventas?\n"
                "• ¿Cuántas filas tiene el archivo?\n"
                "• Muéstrame las primeras 5 filas\n"
                "• ¿Cuál es el valor máximo en edad?\n"
                "• Describe la columna salario\n"
                "• ¿Cuál es la suma total de ingresos?\n"
                "• ¿Cuántos valores únicos hay en categoría?"
            )
            await update.message.reply_text(message)
            return
        
        columnas = df.columns.tolist()[:5]  # Mostrar máximo 5 ejemplos
        ejemplos = []
        
        for col in columnas:
            if df[col].dtype in ['int64', 'float64']:
                ejemplos.append(f"• ¿Cuál es el promedio de {col}?")
            else:
                ejemplos.append(f"• ¿Cuántos valores únicos hay en {col}?")
        
        ejemplos_texto = "\n".join(ejemplos)
        columnas_texto = ", ".join(df.columns.tolist())
        
        message = (
            f"📊 Ejemplos para tu archivo:\n\n{ejemplos_texto}\n\n"
            f"📋 Columnas disponibles: {columnas_texto}\n\n"
            f"💡 También puedes usar /ejecutar [código] para ejecutar código pandas directamente"
        )
        
        await update.message.reply_text(message)
    
    async def info(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /info"""
        if not await self._check_authorization(update):
            return
        
        user_id = update.effective_user.id
        df = self.data_manager.get_dataframe(user_id)
        
        if df is None:
            await update.message.reply_text("📁 No tienes ningún archivo CSV cargado.")
            return
        
        filas, columnas = df.shape
        memoria_mb = df.memory_usage(deep=True).sum() / (1024**2)
        columnas_texto = ", ".join(df.columns.tolist())
        
        message = (
            f"📊 Información del archivo:\n\n"
            f"📏 Dimensiones: {filas:,} filas × {columnas} columnas\n"
            f"💾 Memoria: {memoria_mb:.1f} MB\n"
            f"📋 Columnas: {columnas_texto}\n"
            f"⏰ Expira en: {self._get_ttl_remaining(user_id)} minutos"
        )
        
        await update.message.reply_text(message)
    
    async def limpiar(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /limpiar"""
        if not await self._check_authorization(update):
            return
        
        user_id = update.effective_user.id
        self.data_manager.remove_user(user_id)
        await update.message.reply_text("🧹 Tus datos han sido eliminados correctamente.")
    
    async def estado(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /estado - Solo para usuarios autorizados"""
        if not await self._check_authorization(update):
            return
        
        usuarios_activos = len(self.data_manager.dataframes)
        memoria_uso = self.data_manager._get_memory_usage()
        memoria_sistema = psutil.Process().memory_info().rss / (1024**2)
        
        message = (
            f"⚙️ Estado del Sistema:\n\n"
            f"👥 Usuarios activos: {usuarios_activos}/{MAX_CONCURRENT_USERS}\n"
            f"💾 Memoria datos: {memoria_uso:.1f}/{MAX_MEMORY_MB} MB\n"
            f"🖥️ Memoria sistema: {memoria_sistema:.1f} MB\n"
            f"📊 Cache queries: {len(self.query_cache.cache)} entradas"
        )
        
        await update.message.reply_text(message)
    
    async def recibir_csv(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Maneja recepción de archivos CSV"""
        if not await self._check_authorization(update) or not await self._check_rate_limit(update):
            return
        
        archivo = update.message.document
        user_id = update.effective_user.id
        
        # Validaciones básicas
        if not archivo.file_name.lower().endswith('.csv'):
            await update.message.reply_text("❌ Solo acepto archivos .CSV")
            return
        
        if archivo.file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            await update.message.reply_text(f"❌ Archivo muy grande. Máximo: {MAX_FILE_SIZE_MB}MB")
            return
        
        try:
            # Mensaje de progreso
            progress_msg = await update.message.reply_text("⏳ Descargando y procesando archivo...")
            
            # Descargar archivo
            archivo_telegram = await archivo.get_file()
            archivo_local = f"temp_{user_id}_{int(time.time())}.csv"
            await archivo_telegram.download_to_drive(archivo_local)
            
            # Procesar en background
            df = await self._process_csv_async(archivo_local)
            
            # Validar dimensiones
            if df.shape[0] > MAX_ROWS:
                os.remove(archivo_local)
                await progress_msg.edit_text(f"❌ Archivo con demasiadas filas. Máximo: {MAX_ROWS:,}")
                return
            
            if df.empty:
                os.remove(archivo_local)
                await progress_msg.edit_text("❌ El archivo CSV está vacío")
                return
            
            # Guardar en gestor
            self.data_manager.add_dataframe(user_id, df, archivo_local)
            
            # Crear mensaje SIN MarkdownV2 para evitar problemas de escape
            filas_formateadas = f"{df.shape[0]:,}"
            memoria_formateada = f"{df.memory_usage(deep=True).sum() / (1024**2):.1f}"
            minutos_expira = DATA_TTL_SECONDS // 60
            
            message = (
                f"✅ Archivo procesado exitosamente\n\n"
                f"📊 {filas_formateadas} filas × {df.shape[1]} columnas\n"
                f"💾 {memoria_formateada} MB\n"
                f"⏰ Expira en {minutos_expira} minutos\n\n"
                f"¡Ya puedes hacer preguntas sobre tus datos!"
            )
            
            await progress_msg.edit_text(message)
            
        except Exception as e:
            logger.error(f"Error procesando CSV usuario {user_id}: {e}")
            if 'archivo_local' in locals() and os.path.exists(archivo_local):
                os.remove(archivo_local)
            await update.message.reply_text(f"❌ Error al procesar archivo: {str(e)}")
    
    async def _process_csv_async(self, file_path: str) -> pd.DataFrame:
        """Procesa CSV de manera asíncrona"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._read_csv_safe,
            file_path
        )
    
    def _read_csv_safe(self, file_path: str) -> pd.DataFrame:
        """Lee CSV de manera segura por chunks"""
        try:
            # Intentar lectura normal primero
            df = pd.read_csv(file_path, nrows=1000)  # Sample pequeño
            
            # Si es muy grande, usar chunks
            if os.path.getsize(file_path) > 10 * 1024 * 1024:  # 10MB
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=10000):
                    chunks.append(chunk)
                    if len(chunks) * 10000 > MAX_ROWS:
                        break
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(file_path)
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error al leer CSV: {str(e)}")
    
    async def responder_pregunta(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Responde preguntas sobre el DataFrame"""
        if not await self._check_authorization(update) or not await self._check_rate_limit(update):
            return
        
        user_id = update.effective_user.id
        df = self.data_manager.get_dataframe(user_id)
        
        if df is None:
            await update.message.reply_text("📁 Primero necesitas subir un archivo CSV")
            return
        
        pregunta = update.message.text.strip()
        
        if len(pregunta) < 5:
            await update.message.reply_text("❗ La pregunta es muy corta. Sé más específico")
            return
        
        try:
            # Verificar cache
            cached_result = self.query_cache.get(df.shape, pregunta)
            if cached_result:
                message = f"💾 (Desde cache)\n```\n{cached_result}\n```"
                await update.message.reply_text(message, parse_mode="Markdown")
                return
            
            # Generar respuesta con IA
            progress_msg = await update.message.reply_text("🤔 Analizando tu pregunta...")
            
            respuesta = await self._get_ai_response_async(df, pregunta)
            
            # Guardar en cache y en last_suggested_code
            self.query_cache.set(df.shape, pregunta, respuesta)
            self.last_suggested_code[user_id] = respuesta  # Guardar el último código
            
            message = (
                f"💡 Código sugerido:\n```python\n{respuesta}\n```\n\n"
                f"Usa /ejecutar para ejecutarlo o /ejecutar [otro_código] para otro análisis"
            )
            
            await progress_msg.edit_text(message, parse_mode="Markdown")
            
        except Exception as e:
            logger.error(f"Error generando respuesta para usuario {user_id}: {e}")
            await update.message.reply_text(f"❌ Error al procesar pregunta: {str(e)}")

    async def ejecutar(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /ejecutar - Ejecuta código pandas de forma segura"""
        if not await self._check_authorization(update) or not await self._check_rate_limit(update):
            return
        
        user_id = update.effective_user.id
        df = self.data_manager.get_dataframe(user_id)
        
        if df is None:
            await update.message.reply_text("📁 Primero necesitas subir un archivo CSV")
            return
        
        # Obtener el código - si no hay argumentos, usar el último sugerido
        if not context.args:
            if user_id in self.last_suggested_code:
                codigo = self.last_suggested_code[user_id]
            else:
                await update.message.reply_text(
                    "❗ No hay código sugerido reciente. Primero haz una pregunta o usa:\n"
                    "/ejecutar [código]\n\n"
                    "Ejemplo: /ejecutar df.shape\n"
                    "Ejemplo: /ejecutar df['columna'].mean()"
                )
                return
        else:
            codigo = " ".join(context.args)
        
        try:
            # Validar que el código sea seguro
            codigo_validado = self._validate_safe_code(codigo)
            
            if "# Código detectado como inseguro" in codigo_validado:
                await update.message.reply_text("❌ Código no permitido por seguridad")
                return
            
            # Ejecutar código de forma segura
            progress_msg = await update.message.reply_text("⚙️ Ejecutando código...")
            
            resultado = await self._execute_code_async(df, codigo_validado)
            
            # Formatear resultado
            if isinstance(resultado, (pd.DataFrame, pd.Series)):
                resultado_str = str(resultado)
                if len(resultado_str) > 2000:  # Limitar longitud
                    resultado_str = resultado_str[:2000] + "... (resultado truncado)"
            else:
                resultado_str = str(resultado)
            
            message = f"✅ Resultado:\n```\n{resultado_str}\n```"
            await progress_msg.edit_text(message, parse_mode="Markdown")
            
        except Exception as e:
            logger.error(f"Error ejecutando código usuario {user_id}: {e}")
            await update.message.reply_text(f"❌ Error al ejecutar código: {str(e)}")
    
    async def _execute_code_async(self, df: pd.DataFrame, codigo: str):
        """Ejecuta código de forma asíncrona y segura"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._execute_safe_pandas_code,
            df, codigo
        )
    
    async def _get_ai_response_async(self, df: pd.DataFrame, pregunta: str) -> str:
        """Genera respuesta de IA de manera asíncrona"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._generate_safe_code,
            df, pregunta
        )
    
    def _generate_safe_code(self, df: pd.DataFrame, pregunta: str) -> str:
        """Genera código seguro usando IA"""
        # Crear contexto limitado del DataFrame
        sample_info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'sample': df.head(3).to_dict('records')
        }
        
        prompt = f"""
Tienes un DataFrame de pandas con esta información:
- Filas: {sample_info['shape'][0]}
- Columnas: {sample_info['columns']}
- Tipos: {sample_info['dtypes']}

Pregunta del usuario: {pregunta}

Responde ÚNICAMENTE con una línea de código Python segura usando operaciones básicas de pandas.
Solo usa: .mean(), .sum(), .max(), .min(), .count(), .nunique(), .shape, .head(), .describe()

NO uses: eval(), exec(), import, __

Ejemplo de respuesta válida: df['columna'].mean()
"""
        
        try:
            respuesta_llm = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Eres un asistente que genera código Python seguro para pandas. Solo una línea de código."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-8b-8192",
                max_tokens=100,
                temperature=0.1
            )
            
            code = respuesta_llm.choices[0].message.content.strip()
            # Limpiar código
            code = code.replace('```python', '').replace('```', '').strip()
            
            return self._validate_safe_code(code)
            
        except Exception as e:
            logger.error(f"Error con Groq API: {e}")
            return "df.info()  # Error con IA, mostrando información básica"
    
    def _validate_safe_code(self, code: str) -> str:
        """Valida que el código sea seguro"""
        dangerous_patterns = [
            'import', '__', 'eval', 'exec', 'open', 'file', 'os.',
            'sys.', 'subprocess', 'getattr', 'setattr', 'delattr'
        ]
        
        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                return "df.info()  # Código detectado como inseguro"
        
        return code
    
    def _execute_safe_pandas_code(self, df: pd.DataFrame, codigo: str):
        """Ejecuta código pandas de forma segura"""
        # Crear un namespace limitado y seguro
        safe_globals = {
            '__builtins__': {},  # Sin built-ins peligrosos
            'df': df,
            'pd': pd,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'list': list,
            'dict': dict,
            'sum': sum,
            'max': max,
            'min': min,
        }
        
        safe_locals = {}
        
        try:
            # Ejecutar código en ambiente controlado
            resultado = eval(codigo, safe_globals, safe_locals)
            return resultado
        except Exception as e:
            raise ValueError(f"Error en código: {str(e)}")
    
    def _get_ttl_remaining(self, user_id: int) -> int:
        """Calcula minutos restantes antes de expiración"""
        if user_id not in self.data_manager.timestamps:
            return 0
        
        elapsed = time.time() - self.data_manager.timestamps[user_id]
        remaining = max(0, DATA_TTL_SECONDS - elapsed)
        return int(remaining / 60)

def main():
    """Función principal"""
    try:
        bot = SecureTelegramBot()
        app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        
        # Registrar handlers
        app.add_handler(CommandHandler("start", bot.start))
        app.add_handler(CommandHandler("ayuda", bot.ayuda))
        app.add_handler(CommandHandler("info", bot.info))
        app.add_handler(CommandHandler("limpiar", bot.limpiar))
        app.add_handler(CommandHandler("estado", bot.estado))
        app.add_handler(CommandHandler("ejecutar", bot.ejecutar))
        app.add_handler(MessageHandler(filters.Document.ALL, bot.recibir_csv))
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), bot.responder_pregunta))
        
        # Mensajes de inicio SIN EMOJIS para evitar problemas de encoding
        logger.info("Bot seguro iniciado correctamente")
        print("Bot Telegram CSV Seguro v2.0 iniciado")
        print("Funciones de seguridad activas")
        print("Optimizaciones de rendimiento aplicadas")
        print("Press Ctrl+C para detener...")
        
        app.run_polling()
        
    except KeyboardInterrupt:
        logger.info("Bot detenido por usuario")
    except Exception as e:
        logger.error(f"Error crítico: {e}")
        raise

if __name__ == '__main__':
    main()