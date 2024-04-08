# Import python packages
import streamlit as st    
import pandas as pd
from snowflake.snowpark.context import get_active_session
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

st.set_page_config(layout="wide",page_title="Retail Trends Analysis")

# Define options
options = ['Retail Analysis', 'Retail Sales Prediction']

# Set up sidebar with radio buttons for options
option = st.sidebar.radio('Select Dashboad', options)

if option == 'Retail Analysis':
    # Page Layout
    c1, c2 = st.columns(2)
    with c1:
        st.image('cittabase_logo_bg.jpg', width=250)

    # Get the current credentials
    session = get_active_session()
        
    # Fetch Databases
    result = session.sql('SELECT * FROM RETAIL_SALES.PUBLIC.SALES1;')
    list = result.collect()
    df =  pd.DataFrame(list)
    # st.write(df)
    
    result_clust = session.sql('SELECT * FROM RETAIL_SALES.PUBLIC.SALES_PRED_CLUSTER1;')
    list_clust = result_clust.collect()
    df_clust =  pd.DataFrame(list_clust)
    
    
    ### Data Pre-Processing
    
    ## Datatype validation
    # st.write(df.dtypes)
    # Convert the 'PRICE' column to float type
    df['PRICE'] = df['PRICE'].astype(float)
    # Convert the 'CUSTOMERID' column to string
    df['CUSTOMERID'] = df['CUSTOMERID'].astype(str)
    # Convert the 'INVOICEDATE' column to datetime with the correct format
    df['INVOICEDATE'] = pd.to_datetime(df['INVOICEDATE'], format='%d-%m-%Y %H:%M')
    # Localize the index to UTC
    # df.index = df['INVOICEDATE'].dt.tz_localize('UTC')
    # Convert to the desired timezone (e.g., Asia/Kolkata)
    # desired_timezone = pytz.timezone('Asia/Kolkata')
    # df.index = df.index.tz_convert(desired_timezone)
    # st.write(df.dtypes)
    # st.write(df.dtypes)
    
    
    ## Null Validation
    df['INVOICE'] = df['INVOICE'].replace('', 'Nan')
    df['STOCKCODE'] = df['STOCKCODE'].replace('', 'Nan')
    df['DESCRIPTION'] = df['DESCRIPTION'].replace('', 'Nan')
    df['CUSTOMERID'] = df['CUSTOMERID'].replace('', 'Nan')
    df['COUNTRY'] = df['COUNTRY'].replace('', 'Nan')
    df['QUANTITY'] = df['QUANTITY'].replace('', 0)
    df['PRICE'] = df['PRICE'].replace('', 0)
    df['INVOICEDATE'] = df['INVOICEDATE'].replace('', 'Nan')
    
    
    ## Add Revenue column to the dataframe
    df['REVENUE'] = df['PRICE'] * df['QUANTITY']
    # Convert the 'REVENUE' column to float type
    df['REVENUE'] = df['REVENUE'].astype(float)
    # st.write(df)
    
    ## Add Recent columns to the dataframe
    df1 = df.copy()
    df1['INVOICEDATE'] = pd.to_datetime(df1['INVOICEDATE'])
    # Find the maximum datetime
    max_date = df1['INVOICEDATE'].max()
    # Calculate the difference in days and create a new column
    df1['DAYS'] = (max_date - df1['INVOICEDATE']).dt.days
    # st.write(df1['DAYS'].unique())
    
    
    ## Data Transformation
    # For INVOICE_DATE column
    # Convert 'Date' column to datetime
    df1['INVOICEDATE'] = pd.to_datetime(df1['INVOICEDATE'])
    # Replace year based on conditions
    df1['INVOICEDATE'] = df1['INVOICEDATE'].apply(lambda x: x.replace(year=2022) if x.year == 2009 else x)
    df1['INVOICEDATE'] = df1['INVOICEDATE'].apply(lambda x: x.replace(year=2023) if x.year == 2010 else x)
    df1['INVOICEDATE'] = df1['INVOICEDATE'].apply(lambda x: x.replace(year=2024) if x.year == 2011 else x)
    df1['INVOICEDATE'] = df1['INVOICEDATE'].dt.strftime('%Y-%m-%d %H:%M:%S')
    # st.write(df1)
    
    
    # Categorize Recent Days from DAYS column
    def recentDays_class(days):
                if days <= 50:
                    return '<=50 days'
                elif days >= 51 and days <= 150:
                    return '51-150 days'
                elif days >= 151 and days <= 250:
                    return '151-250 days'
                elif days >= 251 and days <= 350:
                    return '251-350 days'
                elif days >= 351 and days <= 400:
                    return '351-400 days'
                elif days >= 400:
                    return '>=400 days'
                else:
                    return days
            
    df1['RECENT_DAYS'] = df1['DAYS'].apply(recentDays_class)
    # st.write(df1)
    
    
    ## Remove Duplication
    # st.write(df.shape[0])
    # Identify duplicate rows
    duplicate_rows = df1.duplicated()
    # Invert the boolean mask to keep non-duplicate rows
    non_duplicate_df = df1[~duplicate_rows]
    # st.write(non_duplicate_df.shape[0]) ## 1,928 rows were duplicated over 2,000,00
    
    
    ## Prepocessed Data
    prepDf = non_duplicate_df.copy()
    # Sort DataFrame by 'Date' column in ascending order
    prepDf = prepDf.sort_values(by='INVOICEDATE')
    
    head_1, head_2, head_3 = st.columns([20,60, 20])
        
    with head_2:
        # Write directly to the app
        st.markdown("## Retail Sales Performance Analysis")
    
    # For Blank Space
    blank1, blank2 = st.columns(2)
    blank3, blank4 = st.columns(2)
    blank4, blank6 = st.columns(2)
    
    ### Visualizations
    prepDf = prepDf[~prepDf['COUNTRY'].isin(['United Kingdom', 'Unspecified'])]
    
    # st.write(prepDf['INVOICEDATE'].unique())
    
    # Convert 'Date' column to datetime
    prepDf['INVOICEDATE'] = pd.to_datetime(prepDf['INVOICEDATE'])
    
    # page layout
    cola, col1, colb, col2, colc = st.columns([5, 42.5, 2.5, 42.5, 7.5])
    
    ## Revenue By Country
    with col1:
        # Heading
        st.write('##### Overall Revenue By Country')
        sum_rev_cnty = prepDf.groupby('COUNTRY')['REVENUE'].sum().reset_index().sort_values(by='REVENUE', ascending=False)
    
        # Create a bar chart
        fig = px.bar(sum_rev_cnty, y='COUNTRY', x='REVENUE', orientation='h',
                    labels={'COUNTRY': 'Country', 'REVENUE': 'overall Revenue'},
                    text='REVENUE', color= 'COUNTRY',
                    color_discrete_map={'United Arab Emirates': '#0068c9','USA': '#83c9ff','Switzerland': '#ff2b2b','Sweden': '#ffabab','Spain': '#29b09d','Singapore': '#7defa1','Saudi Arabia': '#ff8700','Portugal': '#ffd16a','Poland': '#6d3fc0','Norway': '#d5dae5','Nigeria': '#0068c9','Netherland': '#83c9ff','Lithuania': '#ff2b2b','Lebanon': '#ffabab','Japan': '#29b09d','Italy': '#7defa1','Israel': '#ff8700','Iceland': '#ffd16a','Hong Kong': '#6d3fc0','Greece': '#d5dae5','Germany': '#0068c9','France': '#83c9ff','Finland': '#ff2b2b','EIRE': '#ffabab','Denmark': '#29b09d','Czech Republic': '#7defa1','Cyprus': '#ff8700','Channel Islands': '#ffd16a','Canada': '#6d3fc0','Belgium': '#d5dae5','Bahrain': '#0068c9','Austria': '#83c9ff'})
    
        # Set transparent background
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
            
        # Customize tick and label colors for x-axis and y-axis
        fig.update_xaxes(tickfont=dict(color='#5CA8F1'),  # Change x-axis tick color to blue
                    titlefont=dict(color='#EF7B45'))  # Change x-axis label color to blue
        fig.update_yaxes(tickfont=dict(color='#5CA8F1'),  # Change y-axis tick color to green
                    titlefont=dict(color='#EF7B45'))  # Change y-axis label color to blue
            
        st.plotly_chart(fig, use_container_width=True)
    
    
    ## Count of Customers By Country
    with col2:
        # Heading
        st.write('##### Count of Customers By Country')
        sum_rev_cnty = prepDf.groupby('COUNTRY')['CUSTOMERID'].size().reset_index().sort_values(by='COUNTRY', ascending=False)
        # Create a bar chart
        fig = px.bar(sum_rev_cnty, x='COUNTRY', y='CUSTOMERID',
                    labels={'COUNTRY': 'Country', 'CUSTOMERID': 'Count of Customers'},
                    text='CUSTOMERID', color= 'COUNTRY',
                    color_discrete_map={'United Arab Emirates': '#0068c9','USA': '#83c9ff','Switzerland': '#ff2b2b','Sweden': '#ffabab','Spain': '#29b09d','Singapore': '#7defa1','Saudi Arabia': '#ff8700','Portugal': '#ffd16a','Poland': '#6d3fc0','Norway': '#d5dae5','Nigeria': '#0068c9','Netherland': '#83c9ff','Lithuania': '#ff2b2b','Lebanon': '#ffabab','Japan': '#29b09d','Italy': '#7defa1','Israel': '#ff8700','Iceland': '#ffd16a','Hong Kong': '#6d3fc0','Greece': '#d5dae5','Germany': '#0068c9','France': '#83c9ff','Finland': '#ff2b2b','EIRE': '#ffabab','Denmark': '#29b09d','Czech Republic': '#7defa1','Cyprus': '#ff8700','Channel Islands': '#ffd16a','Canada': '#6d3fc0','Belgium': '#d5dae5','Bahrain': '#0068c9','Austria': '#83c9ff'})
    
        # Set transparent background
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
            
        # Customize tick and label colors for x-axis and y-axis
        fig.update_xaxes(tickfont=dict(color='#5CA8F1'),  # Change x-axis tick color to blue
                    titlefont=dict(color='#EF7B45'))  # Change x-axis label color to blue
        fig.update_yaxes(tickfont=dict(color='#5CA8F1'),  # Change y-axis tick color to green
                    titlefont=dict(color='#EF7B45'))  # Change y-axis label color to blue
            
        st.plotly_chart(fig, use_container_width=True)
        
    
    # page layout
    cold, col3, cole, col4, colf = st.columns([5, 42.5, 5, 42.5, 5])
    
    ## Revenue Over Time Period
    with col3:
        fltDate = prepDf.copy()
        # Heading
        st.write('##### Revenue Trends Over Time')
        fltDate["MONTH_YEAR"] = fltDate["INVOICEDATE"].dt.strftime("%Y %b")
        # st.write(fltDate["MONTH_YEAR"])
        # linechart = pd.DataFrame(fltDate.groupby(fltDate["INVOICEDATE"].dt.strftime("%Y %b"))[
        #                              "REVENUE"].sum())
        sum_rev_time = fltDate.groupby('MONTH_YEAR')['REVENUE'].sum()
        sum_rev_time_df = pd.DataFrame({
                        'Month of Year': sum_rev_time.index,
                        'Revenue': sum_rev_time.values
                    })
        sum_rev_time_df['date'] = pd.to_datetime(sum_rev_time_df['Month of Year'], format='%Y %b')
        sum_rev_time_df['year'] = sum_rev_time_df['date'].dt.year
        sum_rev_time_df['month'] = sum_rev_time_df['date'].dt.month
        
        # Sort the DataFrame by year and then by month
        sum_rev_time_df.sort_values(by=['year', 'month'], inplace=True)
        # st.write(sum_rev_time_df)
        
        fig = px.line(sum_rev_time_df, x='Month of Year', y='Revenue', 
                    labels={'Month of Year': 'Revenue per Month', 'Revenue': 'Total Revenue'},
                    text='Revenue')
        # Set transparent background
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
        
        # Customize tick and label colors for x-axis and y-axis
        fig.update_xaxes(tickfont=dict(color='#5CA8F1'),  # Change x-axis tick color to blue
                        titlefont=dict(color='#EF7B45'))  # Change x-axis label color to blue
        fig.update_yaxes(tickfont=dict(color='#5CA8F1'),  # Change y-axis tick color to green
                        titlefont=dict(color='#EF7B45'))  # Change y-axis label color to blue
        
        fig.update_traces(textposition='top center',  # Move text to the top center of each point
                      textfont=dict(color='red'),  # Customize text font
                      texttemplate='%{text:.1f}',  # Format text
                      hoverinfo='skip')  # Hide hover info to only display text
        st.plotly_chart(fig, use_container_width=True)
    
    
    ## Orders Over Time Period
    with col4:
        # Heading
        st.write('##### Order Trends Over Time')
        sum_ord_time = fltDate.groupby('MONTH_YEAR')['INVOICE'].size()
        sum_ord_time_df = pd.DataFrame({
                        'Month of Year': sum_ord_time.index,
                        'Orders': sum_ord_time.values
                    })
        # st.write(sum_ord_time_df)
        sum_ord_time_df['date'] = pd.to_datetime(sum_ord_time_df['Month of Year'], format='%Y %b')
        sum_ord_time_df['year'] = sum_ord_time_df['date'].dt.year
        sum_ord_time_df['month'] = sum_ord_time_df['date'].dt.month
        
        # Sort the DataFrame by year and then by month
        sum_ord_time_df.sort_values(by=['year', 'month'], inplace=True)
        # st.write(sum_ord_time_df)
        
        fig = px.line(sum_ord_time_df, x='Month of Year', y='Orders', 
                    labels={'Month of Year': 'Orders per Month', 'Orders': 'Count of Orders'},
                    text='Orders')
        # Set transparent background
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
        
        # Customize tick and label colors for x-axis and y-axis
        fig.update_xaxes(tickfont=dict(color='#5CA8F1'),  # Change x-axis tick color to blue
                        titlefont=dict(color='#EF7B45'))  # Change x-axis label color to blue
        fig.update_yaxes(tickfont=dict(color='#5CA8F1'),  # Change y-axis tick color to green
                        titlefont=dict(color='#EF7B45'))  # Change y-axis label color to blue
        
        fig.update_traces(textposition='top center',  # Move text to the top center of each point
                      textfont=dict(color='red'),  # Customize text font
                      texttemplate='%{text:.1f}',  # Format text
                      hoverinfo='skip')  # Hide hover info to only display text
        st.plotly_chart(fig, use_container_width=True)
    
    
    # page layout
    colg, col5, colh, col6, coli = st.columns([2.5, 58, 2, 35, 2.5])
    
    
    ## Top 5 Selling Products
    with col5:
        # Heading
        st.write('##### Top 5 Selling Products By Revenue')
        cnt_desc_rev = prepDf.groupby('DESCRIPTION').agg(Sum_Of_Revenue=('REVENUE', 'sum'), Count_Of_Product=('DESCRIPTION', 'count')).reset_index()
        cnt_desc_rev = cnt_desc_rev.head(5)
        cnt_desc_rev['Sum_Of_Revenue'] = cnt_desc_rev['Sum_Of_Revenue'].round(2) 
        cnt_desc_rev = cnt_desc_rev.sort_values(by='Count_Of_Product',  ascending=False)
        # st.write(cnt_desc_rev)
    
        # Create a bar chart
        fig = px.bar(cnt_desc_rev, x='DESCRIPTION', y='Sum_Of_Revenue',
                    labels={'DESCRIPTION': 'Products', 'Sum_Of_Revenue': 'Sum of Revenue'},
                    text='Sum_Of_Revenue', color= 'DESCRIPTION')
    
        # Set transparent background
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
            
        # Customize tick and label colors for x-axis and y-axis
        fig.update_xaxes(tickfont=dict(color='#5CA8F1'),  # Change x-axis tick color to blue
                    titlefont=dict(color='#EF7B45'))  # Change x-axis label color to blue
        fig.update_yaxes(tickfont=dict(color='#5CA8F1'),  # Change y-axis tick color to green
                    titlefont=dict(color='#EF7B45'))  # Change y-axis label color to blue
            
        st.plotly_chart(fig, use_container_width=True)
    
    ## Distribution of Recent Purchase Over Time
    with col6:
        # Heading
        st.write('##### Recency Of Customer Purchase')
        cnt_recent_purchase = df1.groupby('RECENT_DAYS').size().reset_index(name='count')
        
        fig = px.pie(cnt_recent_purchase, values='count', names='RECENT_DAYS', color='RECENT_DAYS', hole=0.5, 
                    # color_discrete_map={'Germany': '#ff2b2b', 'France': '#0068c9', 'Spain': '#83c9ff'}
                    )
        
        fig.update_layout(legend=dict(title=dict(text='Last Purchase')), plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
        
        # Customize tick and label colors for x-axis and y-axis
        
        fig.update_xaxes(tickfont=dict(color='#5CA8F1'),  # Change x-axis tick color to blue
                        titlefont=dict(color='#EF7B45'))  # Change x-axis label color to blue
        fig.update_yaxes(tickfont=dict(color='#5CA8F1'),  # Change y-axis tick color to green
                        titlefont=dict(color='#EF7B45'))  # Change y-axis label color to blue
        st.plotly_chart(fig, use_container_width=True)
        
    
    # page layout
    desc_col1, desc_col2 = st.columns([5, 95])
    
    with desc_col2:
        st.write('### Customer Segmentation')
    
        # sub-page layout
        # page layout
        sub_desc_col1, sub_desc_col2 = st.columns([5, 95])   
        with sub_desc_col2:
            st.write('**Cluster 0** - Customers who have **low purchases** and **sales** and **recency**.')
            st.write('**Cluster 1** - Customers who have **high purchases** and **sales** and **excellent recency**.')
            st.write('**Cluster 2** - Customers who have **low purchases** and **sales** but with **decent recency**.')
        st.write('')
        
    # page layout
    colg, col7, colh, col8, coli = st.columns([5, 42.5, 5, 42.5, 5])   
    
    with col7:
        # Heading
        st.write('##### Total Sales By Cluster')
    
        df_clust_cnt = df_clust.copy()
        df_clust_cnt['CLUSTER'] = df_clust_cnt['CLUSTER'].astype(str)
        
        sum_sales_cust_cluster = df_clust_cnt.groupby('CLUSTER')['TOTAL_SALES'].sum()
        # sum_sales_cust_cluster = round(sum_sales_cust_cluster,2)
        # st.write(sum_sales_cust_cluster)
    
        sum_sales_cust_cluster_df = pd.DataFrame({
                                'Cluster': sum_sales_cust_cluster.index,
                                'Total Sales': sum_sales_cust_cluster.values
                            })
        # st.write(sum_sales_cust_cluster_df.dtypes)

        sum_sales_cust_cluster_df['Total Sales'] = sum_sales_cust_cluster_df['Total Sales'].astype(float)
        sum_sales_cust_cluster_df['Total Sales'] = round(sum_sales_cust_cluster_df['Total Sales'],2)
        sum_sales_cust_cluster_df['Total Sales'] = sum_sales_cust_cluster_df['Total Sales'].astype(float)

        # sum_sales_cust_cluster_df['Total Sales'] = round(sum_sales_cust_cluster_df['Total Sales'],2)
        # # Create a bar chart
        fig = px.bar(sum_sales_cust_cluster_df, x='Cluster', y='Total Sales',
                    text='Total Sales', color= 'Cluster')
    
        # Set transparent background
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
            
        # Customize tick and label colors for x-axis and y-axis
        fig.update_xaxes(tickfont=dict(color='#5CA8F1'),  # Change x-axis tick color to blue
                    titlefont=dict(color='#EF7B45'))  # Change x-axis label color to blue
        fig.update_yaxes(tickfont=dict(color='#5CA8F1'),  # Change y-axis tick color to green
                    titlefont=dict(color='#EF7B45'))  # Change y-axis label color to blue
            
        st.plotly_chart(fig, use_container_width=True)
    
    
    with col8:
        # Heading
        st.write('##### Number of Customers in Each Cluster')
        cnt_cust_cluster = df_clust_cnt.groupby('CLUSTER').size().reset_index(name='Count')
        # st.write(cnt_cust_cluster)
        # # Create a bar chart
        fig = px.bar(cnt_cust_cluster, x='CLUSTER', y='Count',
                    labels={'Count': 'Count of Customers', 'CLUSTER': 'Cluster'},
                    text='Count', color= 'CLUSTER')
    
        # Set transparent background
        fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
            
        # Customize tick and label colors for x-axis and y-axis
        fig.update_xaxes(tickfont=dict(color='#5CA8F1'),  # Change x-axis tick color to blue
                    titlefont=dict(color='#EF7B45'))  # Change x-axis label color to blue
        fig.update_yaxes(tickfont=dict(color='#5CA8F1'),  # Change y-axis tick color to green
                    titlefont=dict(color='#EF7B45'))  # Change y-axis label color to blue
            
        st.plotly_chart(fig, use_container_width=True)
    
    
    ## Visualization by Prediction
    
    result1 = session.table("RETAIL_SALES.PREDICTION.SALE_PREDICTION1")
    list1 = result1.collect()
    df1_sales =  pd.DataFrame(list1)
     
    result2 = session.table("RETAIL_SALES.PREDICTION.PRODUCT1")
    list2 = result2.collect()
    df2 =  pd.DataFrame(list2)
    # st.write(df2)
    

    fig = go.Figure()
     
    fig.add_trace(go.Scatter(x=df2['INVOICE_DATE'], y=df2['TOTAL_SALES'], mode='lines+markers', name='Historical Sales'))
     
        
    fig.add_trace(go.Scatter(x=df1_sales['TS'], y=df1_sales['FORECAST'], mode='lines', name='Forecast'))
    fig.add_trace(go.Scatter(x=df1_sales['TS'], y=df1_sales['LOWER_BOUND'], fill=None, mode='lines', line_color='green', name='Lower Bound'))
    fig.add_trace(go.Scatter(x=df1_sales['TS'], y=df1_sales['UPPER_BOUND'], fill=None, mode='lines', line_color='maroon', name='Upper Bound'))
    # Layout
    fig.update_layout(xaxis_title='Timestamp',
                          yaxis_title='Value',
                          template='plotly_dark')
    st.write("##### Time Series Forecasting of Sales Using Cortex")
    # plot:
    st.plotly_chart(fig, use_container_width=True)
    
    
if option == 'Retail Sales Prediction':
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    import seaborn as sns
    import matplotlib.pyplot as plt
    import streamlit as st

    # Page Layout
    c1,c2,c3=st.columns([0.2,0.60,0.2])
    with c2:
        # Open the image file
        image = Image.open("ticket_logo.png")

        # Resize the image to the desired height
        desired_height = 600
        desired_width = 2000
        resized_image = image.resize((desired_width, desired_height))
        #Display
        st.image(resized_image, use_column_width=True)
    
    # Get the current credentials
    session = get_active_session()
        
    # Fetch Databases
    result = session.sql('SELECT * FROM RETAIL_SALES.PUBLIC.SALES1;')
    list = result.collect()
    data =  pd.DataFrame(list)
    # st.write(df
    
    # data.StockCode.value_counts()
    
    data['Total_Sales'] = data['QUANTITY'] * data['PRICE']
    
    # sns.scatterplot(x=data.index, y=data['Quantity'])
    # plt.show()
    
    data.isnull().sum()
    
    # Dropping null values
    data.dropna(inplace=True)
    
    # Creating new values
    # Sales
    Sales = data.groupby('CUSTOMERID')['Total_Sales'].sum()
    Sales = Sales.reset_index()
    
    # Invoice
    Invoice = data.groupby('CUSTOMERID')['INVOICE'].count()
    Invoice = Invoice.reset_index()
    
    # max_date
    # data['INVOICEDATE'] = data['INVOICEDATE'].astype('datetime64[ns]')
    data['INVOICEDATE'] = pd.to_datetime(data['INVOICEDATE'], format='%d-%m-%Y %H:%M')
    max_date = data['INVOICEDATE'].max()
    data['Recency'] = max_date - data['INVOICEDATE']
    # data.head()
    
    import datetime as dt
    
    # Recent
    Recent = data.groupby('CUSTOMERID')['Recency'].min()
    Recent = Recent.reset_index()
    
    Recent['Recency'] = Recent['Recency'].dt.days
    
    # Merging data
    First_join = pd.merge(Sales, Invoice, on='CUSTOMERID')
    
    # Final_data
    Cust = pd.merge(First_join, Recent, on='CUSTOMERID')
    # Cust.head()
    
    # Outlier Detection and removal
    col1 = Cust.select_dtypes(include='number')
    
    for i in col1:
        iqr = Cust[i].quantile(0.75) - Cust[i].quantile(0.25)
        Upper_T = Cust[i].quantile(0.75) + (1.5 * iqr)
        Lower_T = Cust[i].quantile(0.25) - (1.5 * iqr)
        Cust[i] =  Cust[i].clip(Upper_T, Lower_T)
    
    # Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    
    scaled = sc.fit_transform(Cust[['INVOICE', 'Total_Sales', 'Recency']])
    s_data = pd.DataFrame(scaled, columns = ['INVOICE', 'Total_Sales', 'Recency'])
    
    # K-means clustering
    from sklearn.cluster import KMeans
    
    scores = []
    for i in range(2, 7):
        kmeans = KMeans(n_clusters=i, max_iter=40, verbose=True).fit(s_data)
        scores.append(kmeans.inertia_)
    
    # plt.plot(range(2,7,1), scores)
    # plt.xticks(ticks=range(2,7))
    # plt.show()
    
    # Silhouette Scores to check clusters credibility
    from sklearn.metrics import silhouette_score
    for i in range(2,7):
        kmeans = KMeans(n_clusters=i, max_iter=40, n_init='auto').fit(s_data)
        cluster_labels = kmeans.labels_
        shilhoutte_avg = silhouette_score(s_data, cluster_labels)
        print(f'Cluster :', i, 'Value :', shilhoutte_avg)
    
    # Final model
    kmeans = KMeans(n_clusters=3, max_iter=35, n_init='auto').fit(s_data)
    cluster_labels = kmeans.labels_
    
    Cust['Cluster'] = cluster_labels
    Cust.head()
    
    # Get centroids
    centroids = kmeans.cluster_centers_
    
    # Cust.to_csv('cluster.csv')
    
    # Distribution of clusters
    # Cust.Cluster.value_counts()
    
    col = ['INVOICE', 'Total_Sales', 'Recency']
    # for i in col:
    #     sns.boxplot(x='Cluster', y=i, data=Cust)
    #     plt.show()
    
    
    from sklearn.metrics import r2_score
    # Function to calculate adjusted R-squared
    def adjusted_r_squared(r_squared, n, k):
        return 1 - ((1 - r_squared) * (n - 1) / (n - k - 1))
                                                                
    ################# Multiple Linear Regression ##########################################
    data['Year'] = data['INVOICEDATE'].dt.year
    data['Month'] = data['INVOICEDATE'].dt.month
    data['Day'] = data['INVOICEDATE'].dt.day
    data['Recency'] = data['Recency'].dt.days
    
    # Altering Variable
    data['STOCKCODE'] = data['STOCKCODE'].astype(str)
    data['STOCKCODE'] = 'D' + data['STOCKCODE']
    
    # Encoding of Variables
    LE = LabelEncoder()
    col = ['STOCKCODE', 'COUNTRY']
    for i in col:
        data[i] = LE.fit_transform(data[i])
    
    # Dropping Unwanted Variables
    data = data.drop(['DESCRIPTION', 'INVOICEDATE', 'QUANTITY', 'PRICE', 'INVOICE', 'CUSTOMERID'], axis=1)
    
    # Capitalize column names for Snowflake purpose
    # data = data.rename(columns={'Country': 'COUNTRY', 'Price': 'PRICE', 'Quantity': 'QUANTITY',
    #                             'StockCode':'STOCKCODE'})
    
    # Splitting of Data
    x = data.drop(['Total_Sales'], axis=1)
    y = data['Total_Sales']
    
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state=0)
    
    # x_train.shape, x_test.shape, y_train.shape, y_test.shape
    
    model = LinearRegression()
    
    # Fitting of model
    model.fit(x_train, y_train)
    
    # Prediction of model
    y_pred = model.predict(x_test)
    
    # Calculate R-squared
    r_squared = r2_score(y_test, y_pred)
    
    # Calculate adjusted R-squared
    n = x_test.shape[0]  
    k = x_test.shape[1]  
    adj_r_squared = adjusted_r_squared(r_squared, n, k)
    
    ################### Snowflake Connection ############
    c1, c2, c3 = st.columns([0.35,0.5,0.1])
    with c2:
        st.header("Sales Prediction")
    
    tab_titles = ['# Single', '# Batch']
    tablet = st.tabs(tab_titles)
    
    with tablet[0]:
        test = session.table("RETAIL_SALES.PREDICTION.SINGLE_TEST")
        result = test.collect()
        test_df = pd.DataFrame(result)
        test_df['INVOICEDATE'] = test_df['INVOICEDATE'].astype('datetime64[ms]')
        test_df['Recency'] = max_date - test_df['INVOICEDATE']
        test_df['Recency'] = test_df['Recency'].dt.days
        test_df['Year'] = test_df['INVOICEDATE'].dt.year
        test_df['Month'] = test_df['INVOICEDATE'].dt.month
        test_df['Day'] = test_df['INVOICEDATE'].dt.day

        test_df['Year'] = test_df['Year'].astype(str)
        test_df['Year'] = test_df['Year'].str.replace(',', '')
        test_df = test_df.drop(['DESCRIPTION', 'INVOICEDATE', 'INVOICE','QUANTITY', 'PRICE'], axis=1)
        
        st.dataframe(test_df)
        test_df = test_df.drop(['CUSTOMERID'], axis=1)
        if st.button("Single Prediction"):
            y_pred = model.predict(test_df)
            y_pred = pd.DataFrame(y_pred)
            y_pred.rename(columns={0: 'TOTAL_SALES'}, inplace=True)
            st.markdown("Predicted Data:")
            st.write('The Total Sale is ' + str(round(y_pred['TOTAL_SALES'][0],2)))       
    
    with tablet[1]:
        test1 = session.table("RETAIL_SALES.PREDICTION.BATCH_TEST")
        result1 = test1.collect()
        test_df1 = pd.DataFrame(result1)

        test_df1['INVOICEDATE'] = pd.to_datetime(test_df1['INVOICEDATE'], format='%d-%m-%Y %H:%M')
        # test_df1['INVOICEDATE'] = test_df1['INVOICEDATE'].astype('datetime64[ms]')
        test_df1['Recency'] = max_date - test_df1['INVOICEDATE']
        test_df1['Recency'] = test_df1['Recency'].dt.days
        test_df1['Year'] = test_df1['INVOICEDATE'].dt.year
        test_df1['Month'] = test_df1['INVOICEDATE'].dt.month
        test_df1['Day'] = test_df1['INVOICEDATE'].dt.day
        test_df1 = test_df1.drop(['DESCRIPTION', 'INVOICEDATE', 'INVOICE','QUANTITY', 'PRICE'], axis=1)

        test_df1['Year'] = test_df1['Year'].astype(str)
        test_df1['Year'] = test_df1['Year'].str.replace(',', '')
        st.dataframe(test_df1)
        test_df1 = test_df1.drop(['CUSTOMERID'], axis=1)
    
        if st.button("Batch Prediction"):
            y_pred = model.predict(test_df1)
            y_pred = pd.DataFrame(y_pred)
            y_pred.rename(columns={0: 'TOTAL_SALES'}, inplace=True)

            a = pd.concat([test_df1, y_pred], axis=1)
            session.write_pandas(a, database="RETAIL_SALES", schema="PREDICTION", table_name="PREDICTED_BATCH", auto_create_table=True, overwrite=True)
            st.write("The predictions are saved to the table PREDICTED_BATCH ")
            data = session.table("RETAIL_SALES.PREDICTION.PREDICTED_BATCH")
            list2 = data.collect()
            data_fr =  pd.DataFrame(list2)
            st.dataframe(data_fr.head(10))
