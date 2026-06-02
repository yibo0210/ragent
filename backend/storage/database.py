"""数据库配置模块（MySQL）"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from backend.config import get_settings

_settings = get_settings()

SQLALCHEMY_DATABASE_URL = _settings.database_url
if not SQLALCHEMY_DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Refusing to start without database configuration.")

# 创建引擎
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_size=10,
    max_overflow=20,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 依赖项
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 初始化表
def init_db():
    Base.metadata.create_all(bind=engine)