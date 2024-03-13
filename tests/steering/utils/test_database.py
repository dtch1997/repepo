import pathlib
from repepo.steering.utils.database import SteeringConfigDatabase
from repepo.steering.utils.helpers import SteeringConfig


def test_create_table():
    db = SteeringConfigDatabase(name="TestTable", db_path="test_db.sqlite")
    assert db.table_exists()
    assert len(db) == 0
    db.delete_table()
    assert not db.table_exists()


def test_delete_table():
    db = SteeringConfigDatabase(name="TestTable", db_path="test_db.sqlite")
    db_path = pathlib.Path("test_db.sqlite")
    assert db.table_exists()
    assert db_path.exists()
    db.delete_table()
    assert not db.table_exists()
    assert not db_path.exists()


def test_insert_config():
    db = SteeringConfigDatabase(name="TestTable", db_path="test_db.sqlite")
    assert len(db) == 0
    db.insert_config(SteeringConfig())
    assert len(db) == 1
    db.delete_table()


def test_get_config_by_eval_hash():
    db = SteeringConfigDatabase(name="TestTable", db_path="test_db.sqlite")
    config = SteeringConfig()
    db.insert_config(config)
    assert db.get_config_by_eval_hash(config.eval_hash) == config
    db.delete_table()


def test_configs_persist_in_database_after_closing_and_reopening():
    db = SteeringConfigDatabase(name="TestTable", db_path="test_db.sqlite")
    db.insert_config(SteeringConfig())
    assert len(db) == 1
    del db

    db = SteeringConfigDatabase(name="TestTable", db_path="test_db.sqlite")
    assert len(db) == 1
    db.delete_table()
