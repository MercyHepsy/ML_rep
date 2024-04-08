import streamlit as st
from snowflake.snowpark import Session
import pandas as pd

session = Session.builder.getOrCreate()

def return_simple_table(session):
    data = session.table("HOSPITAL.READMISSION.BATCH_TEST")
    list = data.collect()
    df =  pd.DataFrame(list)
    df_st = st.dataframe(df)
    return df_st

return_simple_table(session)
