import requests
import json
from pprint import pprint
import pandas as pd
import streamlit as st

######### CONNECTING IICS VIA IICS REST API CONNECTOR##################################
api_url = "https://dm-ap.informaticacloud.com/ma/api/v2/user/login"

header = {
    "Accept":"application/json",
    "Content-Type":"application/json"
}

payload = json.dumps({
"@type": "login",
"username": "prabhakaran.m@cittabase.com",
"password": "Adhavjp@01"
})

response_post = requests.request("POST", api_url,headers=header,data = payload)
response_post.status_code
response_post.json()

################# GETTING TOKEN INFORMATION THROUGH API #############################
url_token = "https://dm-ap.informaticacloud.com/identity-service/api/v1/jwt/Token?client_id=idmc_api&nonce=1234"
header1 = {
    "Accept":"application/json",
    "Content-Type":"application/json",
    "IDS-SESSION-ID":"aK8WktRfLCyb2H9wLAWFyq",
    "cookie":"USER_SESSION=aK8WktRfLCyb2H9wLAWFyq"}

response_get = requests.request("GET", url_token, headers=header1,data = payload)
response_get.status_code
response_get.json()

############### GETTING ASSET DETAILS FROM CGDC THROUGH API #######################
url_ast_det = "https://idmc-api.dm-ap.informaticacloud.com/data360/search/v1/assets?knowledgeQuery=*&segments=all&core.origin=CDGC"

header2 = {
    "Accept":"application/json",
    "Content-Type":"application/json",
    "X-INFA-ORG-ID":"8YNVVHreYiCdDYuGQNON4U",
    "Authorization": "Bearer eyJraWQiOiI3eXpFN0ZIM28zMWdTV3dkak5iOFNOIiwidHlwIjoiSldUIiwiYWxnIjoiRVMyNTYifQ.eyJ1bmlxdWVfaWQiOiIkMmEkMDckQTlsXC83SVRLU09XXC9LcEgxbUxVYU51Y3VnY0VCUUV3VlM4Y0V6c0I3eHVxZnV3SzZGa3V2bSIsInVzZXJfZm5hbWUiOiJQcmFiaGFrYXJhbiIsInVzZXJfbmFtZSI6InByYWJoYWthcmFuLm1AY2l0dGFiYXNlLmNvbSIsImlzcyI6Imh0dHBzOlwvXC9kbS1hcC5pbmZvcm1hdGljYWNsb3VkLmNvbVwvaWRlbnRpdHktc2VydmljZSIsIm5vbmNlIjoiMTIzNCIsInVzZXJfbG5hbWUiOiJNIiwiY2xpZW50X2lkIjoiaWRtY19hcGkiLCJhdWQiOiJjZGxnIiwidXNlcl9vcmdfaWQiOiI4WU5WVkhyZVlpQ2REWXVHUU5PTjRVIiwidXNlcl9pZCI6IjA0VVNmdWwwMk5BZEdYVUVJYThDMHMiLCJvcmdfaWQiOiI4WU5WVkhyZVlpQ2REWXVHUU5PTjRVIiwiZXhwIjoxNzEyMzIxMzUyLCJvcmdfbmFtZSI6IkNJVFRBQkFTRSIsImlhdCI6MTcxMjMxOTU1MiwianRpIjoiM2dJeDNYcWJaa0RlbTU5eTRBSTNhWiJ9.sLYw9Y3W36IcF-V0NgqphVOXxN4U0B0EwXpKIdPZr0DQvE0oVLshDZh_Y-YjFcGp0Vg4EiVFMN6GA26Cj45lAQ"}

response_get = requests.request("POST", url_ast_det,headers=header2,data = payload)
response_get.status_code
pprint(response_get.json())

data = response_get.json()

Dict_values = dict(Origin = [], Name=[], Identity = [])
# Dict_values = dict( Name=[])
for j in range(10):
    Dict_values['Origin'].append(data['hits'][j]['systemAttributes']['core.origin'])
    Dict_values["Name"].append(data['hits'][j]['summary']['core.name'])
    # c.append(data['hits'][j]['summary']['core.location'])
    Dict_values["Identity"].append(data['hits'][j]['core.identity'])

Specs = pd.DataFrame(Dict_values)

if st.button('FETCH_DATA'):
    st.dataframe(Specs)

# a = []
# for j in range(len(data['hits'])):
#     a.append(data['hits'][j]['summary']['core.location'])
