import sqlite3
import pathlib
from repepo.steering.utils.helpers import SteeringConfig


class SteeringConfigDatabase:
    """Database for storing steering configurations and save paths"""

    def __init__(
        self,
        name="SteeringConfig",
        db_path: pathlib.Path | str = "steering_config.sqlite",
    ):
        self.name = name
        if isinstance(db_path, str):
            db_path = pathlib.Path(db_path)
        self.db_path = db_path
        self.con = sqlite3.connect(self.db_path)
        self.cur = self.con.cursor()

        # If table doesn't exist, create it
        if not self.table_exists():
            self.create_table()

    def __del__(self):
        self.con.close()

    def __len__(self):
        self.cur.execute(f"SELECT COUNT(*) FROM {self.name}")
        return self.cur.fetchone()[0]

    def table_exists(self):
        self.cur.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{self.name}'"
        )
        return bool(self.cur.fetchall())

    def create_table(self):
        create_db_command = (
            f"CREATE TABLE {self.name}"
            + """ (
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
        )
        self.cur.execute(create_db_command)
        assert self.table_exists(), "Table creation failed"

    def delete_table(self):
        self.cur.execute(f"DROP TABLE {self.name}")
        self.con.commit()
        self.db_path.unlink()

    def insert_row(self, config: SteeringConfig):
        insert_command = f"""
        INSERT INTO {self.name} (train_hash, eval_hash, model_name, train_dataset, train_split, train_completion_template, test_dataset, test_split, test_completion_template, formatter, aggregator, layer, layer_type, multiplier, patch_generation_tokens_only, skip_first_n_generation_tokens)
        VALUES ('{config.train_hash}', '{config.eval_hash}', '{config.model_name}', '{config.train_dataset}', '{config.train_split}', '{config.train_completion_template}', '{config.test_dataset}', '{config.test_split}', '{config.test_completion_template}', '{config.formatter}', '{config.aggregator}', '{config.layer}', '{config.layer_type}', '{config.multiplier}', '{config.patch_generation_tokens_only}', '{config.skip_first_n_generation_tokens}')
        """
        self.cur.execute(insert_command)
        self.con.commit()

    def get_config_by_eval_hash(self, eval_hash: str) -> SteeringConfig:
        self.cur.execute(f"SELECT * FROM {self.name} WHERE eval_hash='{eval_hash}'")
        row = self.cur.fetchone()
        return SteeringConfig(
            model_name=row[2],
            train_dataset=row[3],
            train_split=row[4],
            train_completion_template=row[5],
            formatter=row[6],
            aggregator=row[7],
            test_dataset=row[8],
            test_split=row[9],
            test_completion_template=row[10],
            layer=row[11],
            layer_type=row[12],
            multiplier=row[13],
            patch_generation_tokens_only=bool(row[14]),
            skip_first_n_generation_tokens=row[15],
        )
