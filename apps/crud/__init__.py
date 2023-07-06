from flask import Flask

import apps.crud.models

# import crud.models

app = Flask(__name__)

if __name__ == "__main__":
    app.run(debug=True)
