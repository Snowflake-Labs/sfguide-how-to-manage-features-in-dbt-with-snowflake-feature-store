import snowflake.snowpark.functions as F
from snowflake.snowpark import Window

zero_if_null = F.builtin("ZEROIFNULL")

def model(dbt, session):
    df_cust_trx_fraud = session.table('CUSTOMER_TRANSACTIONS_FRAUD')

    # Get the Date Information for the Dataset (number of days and start_date)
    date_info = df_cust_trx_fraud.select(
        F.min(F.col("TX_DATETIME")).as_("END_DATE"), F.datediff("DAY", F.col("END_DATE"), F.max(F.col("TX_DATETIME"))).as_("NO_DAYS")
    ).to_pandas()
    days = int(date_info['NO_DAYS'].values[0])
    start_date = str(date_info['END_DATE'].values[0].astype('datetime64[D]'))

    # Create a Snowpark dataframe with one row for each date between the min and max transaction date and save it to a Temp Table
    df_days = session.range(days+1).with_column("TX_DATE", F.to_date(F.dateadd("DAY", F.call_builtin("SEQ4"), F.lit(start_date))))
    df_days.write.save_as_table("df_days", create_temp_table=True)
    df_days = session.table("df_days")

    # Since we aggregate by customer and day and not all customers have transactions for all dates we cross join our date dataframe with our 
    # customer table so each customer will have one row for each date
    distinct_customers = session.sql('SELECT DISTINCT customer_id FROM CUSTOMER_TRANSACTIONS_FRAUD ORDER BY customer_id')
    distinct_customers.write.save_as_table("CUSTOMERS", create_temp_table=True)

    df_customers = session.table("CUSTOMERS").select("CUSTOMER_ID")
    df_cust_day = df_days.join(df_customers)
    # Save dataframe to a temporary table
    df_cust_day.write.save_as_table("df_cust_day", create_temp_table=True)
    df_cust_day = session.table("df_cust_day")

    # Check the min/max transaction and customer-ID ranges
    df_cust_day.select(
        F.col("TX_DATE"),F.col("CUSTOMER_ID")
        ).agg(
            [
                F.min("TX_DATE").as_("MIN_TX_DATE"),F.max("TX_DATE").as_("MAX_TX_DATE"),F.min("CUSTOMER_ID").as_("MIN_CUSTOMER_ID"),
                F.max("CUSTOMER_ID").as_("MAX_CUSTOMER_ID")
            ]
        ).to_pandas()




    df_cust_trx_day = df_cust_trx_fraud.join(
        df_cust_day,
        (df_cust_trx_fraud.col("CUSTOMER_ID") == df_cust_day.col("CUSTOMER_ID")) & (F.to_date(df_cust_trx_fraud.col("TX_DATETIME")) == df_cust_day.col("TX_DATE")),
        "rightouter"
    ).select(
        df_cust_day.col("CUSTOMER_ID").as_("CUSTOMER_ID"), df_cust_day.col("TX_DATE"), zero_if_null(df_cust_trx_fraud.col("TX_AMOUNT")).as_("TX_AMOUNT"),
        F.iff(F.col("TX_AMOUNT") > F.lit(0), F.lit(1), F.lit(0)).as_("NO_TRX")
    ).group_by(F.col("CUSTOMER_ID"), F.col("TX_DATE")).agg([F.sum(F.col("TX_AMOUNT")).as_("TOT_AMOUNT"), F.sum(F.col("NO_TRX")).as_("NO_TRX")])




    # Create the window/partition-by reference and window ranges
    cust_date = Window.partition_by(F.col("customer_id")).orderBy(F.col("TX_DATE"))
    win_7d_cust = cust_date.rowsBetween(-7, -1)
    win_30d_cust = cust_date.rowsBetween(-30, -1)

    df_cust_feat_day = df_cust_trx_day.select(
        F.col("TX_DATE"),F.col("CUSTOMER_ID"),F.col("NO_TRX"),F.col("TOT_AMOUNT"),
        F.lag(F.col("NO_TRX"),1).over(cust_date).as_("CUST_TX_PREV_1"),
        F.sum(F.col("NO_TRX")).over(win_7d_cust).as_("CUST_TX_PREV_7"),
        F.sum(F.col("NO_TRX")).over(win_30d_cust).as_("CUST_TX_PREV_30"),
        F.lag(F.col("TOT_AMOUNT"),1).over(cust_date).as_("CUST_TOT_AMT_PREV_1"),
        F.sum(F.col("TOT_AMOUNT")).over(win_7d_cust).as_("CUST_TOT_AMT_PREV_7"),
        F.sum(F.col("TOT_AMOUNT")).over(win_30d_cust).as_("CUST_TOT_AMT_PREV_30")
    )


    df_date_time_feat = dbt.ref("transactions")

    # Define the Window/partition-by reference
    win_cur_date = Window.partition_by(F.col("PARTITION_KEY")).order_by(F.col("TX_DATETIME")).rangeBetween(Window.unboundedPreceding,Window.currentRow)

    # Create the Customer Behaviour features
    df_cust_behaviour_feat = df_date_time_feat.join(
        df_cust_feat_day, (df_date_time_feat.col("CUSTOMER_ID") == df_cust_feat_day.col("CUSTOMER_ID")) &
        (F.to_date(df_date_time_feat.col("TX_DATETIME")) == df_cust_feat_day.col("TX_DATE"))
    ).with_column(
        "PARTITION_KEY", F.concat(df_date_time_feat.col("CUSTOMER_ID"), F.to_date(df_date_time_feat.col("TX_DATETIME")))
    ).with_columns(
        ["CUR_DAY_TRX", "CUR_DAY_AMT"], [F.count(df_date_time_feat.col("CUSTOMER_ID")).over(win_cur_date), F.sum(df_date_time_feat.col("TX_AMOUNT")).over(win_cur_date)]
    ).select(
        df_date_time_feat.col("CUSTOMER_ID").as_("CUSTOMER_ID"), df_date_time_feat.col("TX_DATETIME").as_("EVENT_TIMESTAMP"),
        (zero_if_null(df_cust_feat_day.col("CUST_TX_PREV_1")) + F.col("CUR_DAY_TRX")).as_("CUST_CNT_TX_1"),
        (zero_if_null(df_cust_feat_day.col("CUST_TOT_AMT_PREV_1")) + F.col("CUR_DAY_AMT")).as_("CUST_TOT_AMOUNT_1"),
        (zero_if_null((df_cust_feat_day.col("CUST_TOT_AMT_PREV_1")) + F.col("CUR_DAY_AMT")) / F.col("CUST_CNT_TX_1")).as_("CUST_AVG_AMOUNT_1"),
        (zero_if_null(df_cust_feat_day.col("CUST_TX_PREV_7")) + F.col("CUR_DAY_TRX")).as_("CUST_CNT_TX_7"),
        (zero_if_null(df_cust_feat_day.col("CUST_TOT_AMT_PREV_7")) + F.col("CUR_DAY_AMT")).as_("CUST_TOT_AMOUNT_7"),
        (zero_if_null((df_cust_feat_day.col("CUST_TOT_AMT_PREV_7")) + F.col("CUR_DAY_AMT")) / F.col("CUST_CNT_TX_7")).as_("CUST_AVG_AMOUNT_7"),
        (zero_if_null(df_cust_feat_day.col("CUST_TX_PREV_30")) + F.col("CUR_DAY_TRX")).as_("CUST_CNT_TX_30"),
        (zero_if_null(df_cust_feat_day.col("CUST_TOT_AMT_PREV_30")) + F.col("CUR_DAY_AMT")).as_("CUST_TOT_AMOUNT_30"),
        (zero_if_null((df_cust_feat_day.col("CUST_TOT_AMT_PREV_30")) + F.col("CUR_DAY_AMT")) / F.col("CUST_CNT_TX_30")).as_("CUST_AVG_AMOUNT_30")
    ) 

    df_cust_behaviour_feat = df_cust_behaviour_feat.select(
        F.col("CUSTOMER_ID"), F.col("EVENT_TIMESTAMP"), F.col("CUST_CNT_TX_1"), F.col("CUST_TOT_AMOUNT_1"), F.col("CUST_AVG_AMOUNT_1"), F.col("CUST_CNT_TX_7"),
        F.col("CUST_TOT_AMOUNT_7"), F.col("CUST_AVG_AMOUNT_7"), F.col("CUST_CNT_TX_30"), F.col("CUST_TOT_AMOUNT_30"), F.col("CUST_AVG_AMOUNT_30")
    )

    return df_cust_behaviour_feat