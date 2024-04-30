import pathlib
from repepo.experiments.cross_steering_result_db import CrossSteeringResultDatabase


def test_create_table():
    db = CrossSteeringResultDatabase(name="TestTable", db_path="test_db.sqlite")
    assert db.table_exists()
    assert len(db) == 0
    db.delete_table()
    assert not db.table_exists()


def test_delete_table():
    db = CrossSteeringResultDatabase(name="TestTable", db_path="test_db.sqlite")
    db_path = pathlib.Path("test_db.sqlite")
    assert db.table_exists()
    assert db_path.exists()
    db.delete_table()
    assert not db.table_exists()
    assert not db_path.exists()


def test_add():
    db = CrossSteeringResultDatabase(name="TestTable", db_path="test_db.sqlite")
    assert len(db) == 0
    db.add(
        steering_vector_dataset_name="test",
        steering_vector_dataset_variant="test",
        steering_vector_multiplier=1.0,
        test_dataset_name="test",
        test_dataset_variant="test",
        test_example_id=0,
        test_example_positive_text="test",
        test_example_negative_text="test",
        test_example_logit_diff=0.0,
        test_example_pos_prob=0.0,
    )
    db.delete_table()


def test_data_persist_in_database_after_closing_and_reopening():
    db = CrossSteeringResultDatabase(name="TestTable", db_path="test_db.sqlite")
    db.add(
        steering_vector_dataset_name="test",
        steering_vector_dataset_variant="test",
        steering_vector_multiplier=1.0,
        test_dataset_name="test",
        test_dataset_variant="test",
        test_example_id=0,
        test_example_positive_text="test",
        test_example_negative_text="test",
        test_example_logit_diff=0.0,
        test_example_pos_prob=0.0,
    )
    assert len(db) == 1
    del db

    db = CrossSteeringResultDatabase(name="TestTable", db_path="test_db.sqlite")
    assert len(db) == 1
    db.delete_table()
