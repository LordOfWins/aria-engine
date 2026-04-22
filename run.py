"""ARIA Engine - Server Runner"""

import uvicorn
from aria.core.config import get_config


def main() -> None:
    config = get_config()
    uvicorn.run(
        "aria.api.app:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.env.value == "development",
        log_level=config.api.log_level.lower(),
    )


if __name__ == "__main__":
    main()
