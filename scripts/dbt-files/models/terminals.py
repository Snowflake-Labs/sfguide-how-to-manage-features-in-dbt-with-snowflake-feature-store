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

    # Since we aggregate by terminal and day and not all terminals have transactions for all dates we cross join our date dataframe with our terminal table so each terminal will have one row for each date
    distinct_terminals = session.sql('SELECT DISTINCT terminal_id FROM CUSTOMER_TRANSACTIONS_FRAUD ORDER BY terminal_id')
    distinct_terminals.write.save_as_table("TERMINALS", create_temp_table=True)

    df_terminals = session.table("TERMINALS").select("TERMINAL_ID")
    df_term_day = df_days.join(df_terminals)

    # Aggregate number of transactions and amount by terminal and date, for dates where a terminal do not have any transactions we add a 0
    df_term_trx_by_day = df_cust_trx_fraud.join(
        df_term_day,
        (df_cust_trx_fraud.col("TERMINAL_ID") == df_term_day.col("TERMINAL_ID")) & (F.to_date(df_cust_trx_fraud.col("TX_DATETIME")) == df_term_day.col("TX_DATE")),
        "rightouter"
    ).select(
        df_term_day.col("TERMINAL_ID").as_("TERMINAL_ID"),
        df_term_day.col("TX_DATE"),
        zero_if_null(df_cust_trx_fraud.col("TX_FRAUD")).as_("NB_FRAUD"),
        F.when(F.is_null(df_cust_trx_fraud.col("TX_FRAUD")), F.lit(0)).otherwise(F.lit(1)).as_("NO_TRX")
    ) .groupBy(
        F.col("TERMINAL_ID"), F.col("TX_DATE")
    ).agg(
        [F.sum(F.col("NB_FRAUD")).as_("NB_FRAUD"), F.sum(F.col("NO_TRX")).as_("NO_TRX")]
    )

    # Aggregate by our windows.
    term_date = Window.partitionBy(F.col("TERMINAL_ID")).orderBy(F.col("TX_DATE"))
    win_delay = term_date.rowsBetween(-7, -1) 
    win_1d_term = term_date.rowsBetween(-8, -1) # We need to add the Delay period to our windows
    win_7d_term = term_date.rowsBetween(-14, -1)
    win_30d_term = term_date.rowsBetween(-37, -1)

    # Define the terminal daily features
    df_term_feat_day = df_term_trx_by_day.select(
        F.col("TX_DATE"),F.col("TERMINAL_ID").as_("TID"),
        F.col("NO_TRX"), F.col("NB_FRAUD"),
        F.sum(F.col("NB_FRAUD")).over(win_delay).as_("NB_FRAUD_DELAY"),
        F.sum(F.col("NO_TRX")).over(win_delay).as_("NB_TX_DELAY"),
        F.sum(F.col("NO_TRX")).over(win_1d_term).as_("NB_TX_DELAY_WINDOW_1"),
        F.sum(F.col("NO_TRX")).over(win_1d_term).as_("NB_TX_DELAY_WINDOW_7"),
        F.sum(F.col("NO_TRX")).over(win_30d_term).as_("NB_TX_DELAY_WINDOW_30"),
        F.sum(F.col("NB_FRAUD")).over(win_1d_term).as_("NB_FRAUD_DELAY_WINDOW_1"),
        F.sum(F.col("NB_FRAUD")).over(win_1d_term).as_("NB_FRAUD_DELAY_WINDOW_7"),
        F.sum(F.col("NB_FRAUD")).over(win_30d_term).as_("NB_FRAUD_DELAY_WINDOW_30")
    )

    win_cur_date = Window.partition_by(F.col("PARTITION_KEY")).order_by(F.col("TX_DATETIME")).rangeBetween(Window.unboundedPreceding,Window.currentRow)

    # Finally, combine all the Terminal features into a single table
    df_term_behaviour_feat = df_cust_trx_fraud.join(
        df_term_feat_day,
        (df_cust_trx_fraud.col("TERMINAL_ID") == df_term_feat_day.col("TID")) & (F.to_date(df_cust_trx_fraud.col("TX_DATETIME")) == df_term_feat_day.col("TX_DATE"))
    ).with_columns(
        ["PARTITION_KEY", "CUR_DAY_TRX", "CUR_DAY_FRAUD"],
        [
            F.concat(df_cust_trx_fraud.col("TERMINAL_ID"), F.to_date(df_cust_trx_fraud.col("TX_DATETIME"))),
            F.count(df_cust_trx_fraud.col("TERMINAL_ID")).over(win_cur_date),
            F.sum(df_cust_trx_fraud.col("TX_FRAUD")).over(win_cur_date)
        ]
    ).with_columns(
        [
            "NB_TX_DELAY", "NB_FRAUD_DELAY", "NB_TX_DELAY_WINDOW_1", "NB_FRAUD_DELAY_WINDOW_1", "NB_TX_DELAY_WINDOW_7",
            "NB_FRAUD_DELAY_WINDOW_7", "NB_TX_DELAY_WINDOW_30", "NB_FRAUD_DELAY_WINDOW_30"
        ],
        [
            df_term_feat_day.col("NB_TX_DELAY") + F.col("CUR_DAY_TRX"), F.col("NB_FRAUD_DELAY") +  F.col("CUR_DAY_FRAUD"), F.col("NB_TX_DELAY_WINDOW_1") + F.col("CUR_DAY_TRX"),
            F.col("NB_FRAUD_DELAY_WINDOW_1") + F.col("CUR_DAY_FRAUD"), F.col("NB_TX_DELAY_WINDOW_7") + F.col("CUR_DAY_TRX"), F.col("NB_FRAUD_DELAY_WINDOW_7") + F.col("CUR_DAY_FRAUD"),
            F.col("NB_TX_DELAY_WINDOW_30") + F.col("CUR_DAY_TRX"), F.col("NB_FRAUD_DELAY_WINDOW_30") + F.col("CUR_DAY_FRAUD")
        ]
    ).select(
        df_cust_trx_fraud.col("TERMINAL_ID").as_("TERMINAL_ID"), df_cust_trx_fraud.col("TX_DATETIME").as_("EVENT_TIMESTAMP"),
        (F.col("NB_TX_DELAY_WINDOW_1") - F.col("NB_TX_DELAY")).as_("NB_TX_WINDOW_1"),
        F.iff(F.col("NB_FRAUD_DELAY_WINDOW_1") - F.col("NB_FRAUD_DELAY") > 0, (F.col("NB_FRAUD_DELAY_WINDOW_1") - F.col("NB_FRAUD_DELAY")) / F.col("NB_TX_WINDOW_1"), F.lit(0)).as_("TERM_RISK_1"),
        (F.col("NB_TX_DELAY_WINDOW_7") - F.col("NB_TX_DELAY")).as_("NB_TX_WINDOW_7"),
        F.iff(F.col("NB_FRAUD_DELAY_WINDOW_7") - F.col("NB_FRAUD_DELAY") > 0, (F.col("NB_FRAUD_DELAY_WINDOW_7") - F.col("NB_FRAUD_DELAY")) / F.col("NB_TX_WINDOW_7"), F.lit(0)).as_("TERM_RISK_7"),
        (F.col("NB_TX_DELAY_WINDOW_30") - F.col("NB_TX_DELAY")).as_("NB_TX_WINDOW_30"),
        F.iff(F.col("NB_FRAUD_DELAY_WINDOW_30") - F.col("NB_FRAUD_DELAY") > 0, (F.col("NB_FRAUD_DELAY_WINDOW_30") - F.col("NB_FRAUD_DELAY"))  / F.col("NB_TX_WINDOW_30"), F.lit(0)).as_("TERM_RISK_30")
    )

    return df_term_behaviour_feat
