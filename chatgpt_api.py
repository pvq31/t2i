import http.client
import json

conn = http.client.HTTPSConnection("api.chatanywhere.tech")
payload = json.dumps({
   "model": "gpt-5.2",
   "input": "你好,你是什么模型,什么版本"
})
headers = {
   'Authorization': 'Bearer sk-4vGkTSWP4WmCIQaiY1fthMExZHJL5aDl2Ad8QmtZiGcJYJzS',
   'Content-Type': 'application/json'
}
conn.request("POST", "/v1/responses", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))