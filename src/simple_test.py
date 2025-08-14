import modal

app = modal.App("simple-test")

@app.function()
@modal.fastapi_endpoint(method="GET")
def hello():
    return {"message": "Hello from Modal!"}