from db import get_db

class User:
    def __init__(self, user_email, risk_level):
        self.user_email = user_email
        self.risk_level = risk_level
    @staticmethod
    def get(user_email):
        db = get_db()
        row = db.execute(
            "SELECT * FROM user WHERE user_email = ?", (user_email,)
        ).fetchone()

        if not row:
            return None

        user = User(
                        user_email=row[1],
                        risk_level=row[2],
                       )
        return user

    @staticmethod
    def create(user_email, risk_level):

        db = get_db()
        db.execute(
            "INSERT INTO user (user_email, risk_level)"
            "VALUES (?, ?)", (user_email, risk_level),
        )
        db.commit()

    @staticmethod
    def set_risk(email, risk):
        db = get_db()
        
        db.execute(
            "UPDATE user SET risk_level = ? WHERE user_email = ?", (risk, email)
        ).fetchone()

        db.commit()