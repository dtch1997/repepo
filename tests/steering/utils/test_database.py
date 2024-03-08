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


def test_insert_row():
    db = SteeringConfigDatabase(name="TestTable", db_path="test_db.sqlite")
    assert len(db) == 0
    db.insert_row(SteeringConfig())
    assert len(db) == 1
    db.delete_table()


def test_persistence():
    """Test that entries remain in the database after database is closed and reopened"""
    db = SteeringConfigDatabase(name="TestTable", db_path="test_db.sqlite")
    db.insert_row(SteeringConfig())
    assert len(db) == 1
    del db

    db = SteeringConfigDatabase(name="TestTable", db_path="test_db.sqlite")
    assert len(db) == 1
    db.delete_table()
