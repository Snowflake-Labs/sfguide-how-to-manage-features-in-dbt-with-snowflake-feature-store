import snowflake.snowpark.functions as F

def model(dbt, session):
    df_cust_trx_fraud = session.table('CUSTOMER_TRANSACTIONS_FRAUD')

    df_date_time_feat = df_cust_trx_fraud.with_columns(
        [ "CT_DATETIME", "TX_DURING_WEEKEND", "TX_DURING_NIGHT"],
        [
            F.current_timestamp(),
            F.iff((F.dayofweek(F.col("TX_DATETIME")) == F.lit(6)) | (F.dayofweek(F.col("TX_DATETIME")) == F.lit(0)), F.lit(1), F.lit(0)),
            F.iff((F.hour(F.col("TX_DATETIME")) < F.lit(6)) | (F.hour(F.col("TX_DATETIME")) > F.lit(20)), F.lit(1), F.lit(0))
        ]
    ).select(
        F.col("TRANSACTION_ID"), F.col("CUSTOMER_ID"), F.col("TERMINAL_ID"), F.col("TX_DATETIME").as_("TX_DATETIME"), F.col("TX_DATETIME").as_("EVENT_TIMESTAMP"),
        F.col("CT_DATETIME").as_("CT_DATETIME"), F.col("TX_AMOUNT"), F.col("TX_TIME_SECONDS"), F.col("TX_TIME_DAYS"), F.col("TX_FRAUD"), F.col("TX_FRAUD_SCENARIO"),
        F.col("TX_DURING_WEEKEND"), F.col("TX_DURING_NIGHT")
    )
    
    return df_date_time_feat