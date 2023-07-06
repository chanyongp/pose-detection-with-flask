from flask import Blueprint, render_template

from apps.app import db
from apps.crud.models import User
from apps.detector.models import UserImage

dt = Blueprint("detector", __name__, template_folder="templates")


@dt.route("/")
def index():
    user_images = (
        db.session.query(User, UserImage)
        .join(UserImage)
        .filter(User.id == UserImage.user_id)
        .all()
    )

    return render_template("detector/index.html", user_images=user_images)


"""
@dt.route("/images/<path:filename>")
def image_file(filename):
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)

"""


@dt.errorhandler(404)
def page_not_found(e):
    return render_template("detector/404.html"), 404
