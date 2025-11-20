import logging
import sys
import json
from logging.handlers import TimedRotatingFileHandler

# Define log format
FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LOG_FILE = "app.log"  # Log file name


class CustomLogger(logging.Logger):
    """
    自定义 Logger 类，增加了 log_struct 方法
    以支持结构化 (JSON) 日志记录。
    """

    def log_struct(self, data, level=logging.INFO):
        """
        记录一个结构化的 JSON 消息。
        这就是 score.py 试图调用的方法。
        """
        if not isinstance(data, dict):
            self.warning(f"log_struct called with non-dict data: {data}")
            return

        try:
            # (我们假设 'josn_obj' 在原始代码中是 'json_obj' 的拼写错误)
            message = json.dumps(data)
            self.log(level, message)
        except TypeError as e:
            self.error(f"Failed to serialize structured log: {e}")


# 告知 Python 的 logging 模块使用我们的 CustomLogger 类
logging.setLoggerClass(CustomLogger)


def get_console_handler():
    """Get console handler"""
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler():
    """Get file handler"""
    file_handler = TimedRotatingFileHandler(
        LOG_FILE, when="midnight", backupCount=5
    )
    file_handler.setFormatter(FORMATTER)
    return file_handler


def setup_logging(log_level=logging.INFO):
    """Setup logging"""
    # 获取 root logger (这将是一个 CustomLogger 实例)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Add handlers if they don't exist
    if not root_logger.hasHandlers():
        root_logger.addHandler(get_console_handler())
        root_logger.addHandler(get_file_handler())

    # Set higher level for httpx to avoid spam
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return root_logger


# Initial setup
setup_logging()

# ==============================================================================
# 导出 'logger' (实例) 和 'CustomLogger' (类)
# ==============================================================================
# 这将创建一个 CustomLogger 实例，其他模块可以导入它
logger = logging.getLogger(__name__)

# CustomLogger 类定义在上面，所以 'from src.logger import CustomLogger' 也会成功