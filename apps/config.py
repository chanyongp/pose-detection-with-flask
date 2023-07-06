from pathlib import Path

basedir = Path(__file__).parent.parent


class BaseConfig:
    SECRET_KEY = "OqF4ZgbYK1AjGeL"
    WTF_CSRF_SECRET_KEY = "tvTOSycXUoErbH6"
    # UPLOAD_FOLDER = str(Path(basedir, "apps", "images"))


class LocalConfig(BaseConfig):
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{basedir / 'local.sqlite'}"
    SQLALCHEMY_TRACK_MODIFICATION = False
    SQLALCHEMY_ECHO = True


class TestingConfig(BaseConfig):
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{basedir / 'testing.sqlite'}"
    SQLALCHEMY_TRACK_MODIFICATION = False
    WTF_CSRF_ENABLED = False


config = {"testing": TestingConfig, "local": LocalConfig}
