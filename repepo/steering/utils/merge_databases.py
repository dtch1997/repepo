import sqlite3
import pathlib
import simple_parsing

if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_argument("--db-path-source", type=str, required=True)
    parser.add_argument("--db-path-target", type=str, required=True)
    args = parser.parse_args()

    db_path_source = pathlib.Path(args.db_path_source)
    db_path_target = pathlib.Path(args.db_path_target)

    # Connect to the first database
    con = sqlite3.connect(db_path_target)

    # Attach the second database to the first database connection
    con.execute(f"ATTACH DATABASE '{db_path_source.absolute()}' AS db2")

    # Insert rows from the SteeringConfig table in the second database into the first,
    # only if the eval_hash does not already exist in the first database
    con.execute(
        """
    INSERT INTO SteeringConfig (eval_hash, train_hash, model_name, train_dataset, train_split, train_completion_template, formatter, aggregator, test_dataset, test_split, test_completion_template, layer, layer_type, multiplier, patch_generation_tokens_only, skip_first_n_generation_tokens)
    SELECT eval_hash, train_hash, model_name, train_dataset, train_split, train_completion_template, formatter, aggregator, test_dataset, test_split, test_completion_template, layer, layer_type, multiplier, patch_generation_tokens_only, skip_first_n_generation_tokens
    FROM db2.SteeringConfig
    WHERE NOT EXISTS (
        SELECT 1 FROM SteeringConfig WHERE eval_hash = db2.SteeringConfig.eval_hash
    )
    """
    )

    print("Merging SteeringConfig table done")

    # Commit the transaction and close the connection
    con.commit()
    con.close()
