from flask import Flask, render_template, request
from model import predict, teams

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def home():

    result=None

    if request.method=="POST":

        h=request.form["home"]
        a=request.form["away"]

        result=predict(h,a)

    return render_template("index.html",teams=teams,result=result)

if __name__=="__main__":
    app.run(debug=True)
