from dataclasses import dataclass, field
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

@dataclass
class NEO4JConfig:
    """neo4j 导入流程配置"""
    entity_name_collection: str = field(
        default_factory=lambda: os.getenv("ENTITY_NAME_COLLECTION", "")
    )
    neo4j_uri: str = field(
        default_factory=lambda: os.getenv("NEO4J_URI", "")
    )
    neo4j_username: str = field(
        default_factory=lambda: os.getenv("NEO4J_USERNAME", "")
    )
    neo4j_password: str = field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", "")
    )
    neo4j_database: str = field(
        default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j")
    )

    @classmethod
    def from_env(cls) -> "NEO4JConfig":
        """从环境变量加载配置"""
        return cls()

# ==================== 全局单例 ====================
_config: Optional[NEO4JConfig] = None

def get_config() -> NEO4JConfig:
    """获取配置单例"""
    global _config
    if _config is None:
        _config = NEO4JConfig.from_env()
    return _config