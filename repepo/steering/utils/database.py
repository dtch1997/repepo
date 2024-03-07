import sqlite3
from repepo.steering.utils.helpers import SteeringConfig


class SteeringConfigDatabase:
    """Database for storing steering configurations and save paths"""

    def __init__(self):
        self.con = sqlite3.connect("steering_config.db")
        self.cur = self.con.cursor()

        # If table doesn't exist, create it
        if not self.table_exists():
            self.create_table()

    def table_exists(self):
        self.cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='SteeringConfig'"
        )
        return bool(self.cur.fetchall())

    def create_table(self):
        create_db_command = """
        CREATE TABLE SteeringConfig (
            eval_hash VARCHAR(255) PRIMARY KEY,
            train_hash VARCHAR(255) NOT NULL,
            model_name VARCHAR(255) NOT NULL DEFAULT 'meta-llama/Llama-2-7b-chat-hf',
            train_dataset VARCHAR(255) NOT NULL DEFAULT 'sycophancy_train',
            train_split VARCHAR(255) NOT NULL DEFAULT '0%:+10',
            train_completion_template VARCHAR(255) NOT NULL DEFAULT '{prompt} {response}',
            formatter VARCHAR(255) NOT NULL DEFAULT 'identity-formatter',
            aggregator VARCHAR(255) NOT NULL DEFAULT 'mean',
            test_dataset VARCHAR(255) NOT NULL DEFAULT 'sycophancy_train',
            test_split VARCHAR(255) NOT NULL DEFAULT '40%+10',
            test_completion_template VARCHAR(255) NOT NULL DEFAULT '{prompt} {response}',
            layer INT NOT NULL DEFAULT 0,
            layer_type VARCHAR(255) NOT NULL DEFAULT 'decoder_block',
            multiplier FLOAT NOT NULL DEFAULT 0,
            patch_generation_tokens_only BOOLEAN NOT NULL DEFAULT TRUE,
            skip_first_n_generation_tokens INT NOT NULL DEFAULT 0
        )
        """
        self.cur.execute(create_db_command)
        assert self.table_exists(), "Table creation failed"

    def insert_row(self, config: SteeringConfig):
        insert_command = f"""
        INSERT INTO SteeringConfig (train_hash, eval_hash, model_name, train_dataset, train_split, train_completion_template, test_dataset, test_split, test_completion_template, formatter, aggregator, layer, layer_type, multiplier, patch_generation_tokens_only, skip_first_n_generation_tokens)
        VALUES ('{config.train_hash}', '{config.eval_hash}', '{config.model_name}', '{config.train_dataset}', '{config.train_split}', '{config.train_completion_template}', '{config.test_dataset}', '{config.test_split}', '{config.test_completion_template}', '{config.formatter}', '{config.aggregator}', '{config.layer}', '{config.layer_type}', '{config.multiplier}', '{config.patch_generation_tokens_only}', '{config.skip_first_n_generation_tokens}')
        """
        self.cur.execute(insert_command)
