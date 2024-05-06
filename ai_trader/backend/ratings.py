from db import get_db

class Ratings:
    def __init__(self, review_dict):
        self.email=review_dict['email']
        self.course_code = review_dict['course_code']
        self.couse_name = review_dict['course_name']
        self.professor_name = review_dict['professor_name']
        self.rating_prof = review_dict['rating_prof']
        self.rate_cvn_quality = review_dict['rate_cvn_quality']
        self.rate_cvn_resources = review_dict['rate_cvn_resources']
        self.rate_cvn_difficulty = review_dict['rate_cvn_difficulty']
        self.tag_1 = review_dict['tag_1']
        self.tag_2 = review_dict['tag_2']
        self.tag_3 = review_dict['tag_3']
        self.user_review = review_dict['user_review']
        self.flag_count = review_dict['flag_count']

    @staticmethod
    def get(rating_id):
        db = get_db()
        rating = db.execute(
            "SELECT * FROM ratings WHERE id = ?", (rating_id,)
        ).fetchone()
        if not rating:
            return None

        rating = Ratings(
                        email=rating[0],
                        course_code=rating[1],
                        course_name=rating[2],
                        professor_name=rating[3],
                        rating_prof=rating[4],
                        rate_cvn_quality=rating[5],
                        rate_cvn_resources=rating[6],
                        rate_cvn_difficulty=rating[7],
                        tag_1=rating[8],
                        tag_2=rating[9],
                        tag_3=rating[10],
                        user_review=rating[11],
                        flag_count=rating[12]
                       )
        return rating

    @staticmethod
    def create(email, course_code, course_name, professor_name, rating_prof,
               rate_cvn_quality, rate_cvn_resources, rate_cvn_difficulty,
               tag_1, tag_2, tag_3, user_review, flag_count, like_count, dislike_count, date):
        db = get_db()

        db.execute(
            "INSERT INTO ratings (email, course_code, course_name, \
            professor_name, rating_prof, rate_cvn_quality, \
            rate_cvn_resources, rate_cvn_difficulty, tag_1, tag_2, tag_3, user_review, flag_count, like_count, dislike_count, date)"

            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (email, course_code, course_name, professor_name, rating_prof,
               rate_cvn_quality, rate_cvn_resources, rate_cvn_difficulty,
               tag_1, tag_2, tag_3, user_review, flag_count, like_count, dislike_count, date),
        )
        db.commit()

    @staticmethod
    def search(search_string):
        without_parentheses = search_string.replace('(', '').replace(')', '')
        query_list = without_parentheses.split()

        result_list = []
        db = get_db()
        print(query_list)

        for query_string in query_list:
            results = db.execute(
                "SELECT * FROM ratings WHERE course_code LIKE ? OR course_name LIKE ? OR \
                    professor_name LIKE ?", ('%' + query_string + '%', '%' + query_string + '%', '%' + query_string + '%')
            ).fetchall()

            for rating in results:
                result = {
                    'id':rating[0],
                    'courseCode': rating[2],
                    'courseName': rating[3],
                    'name': rating[4],
                    'ratings': [rating[5], rating[6], rating[7], rating[8]],
                }

                if result not in result_list:
                    result_list.append(result)
        
        return result_list
    
    @staticmethod
    def my_reviews(email):
        result_list = []
        db = get_db()
        results = db.execute(
            "SELECT * FROM ratings WHERE email = ?", (email,)
        ).fetchall()
        if not results:
            return None

        for rating in results:
            result = {
                'id':rating[0],
                'courseCode': rating[2],
                'courseName': rating[3],
                'name': rating[4],
                'ratings': [rating[5], rating[6], rating[7], rating[8]],
                'tags': [rating[9], rating[10], rating[11]],
                'review': rating[12],
                "date": rating[16],
            }

            result_list.append(result)
        
        return result_list
    
    @staticmethod
    def search_prof_reviews(query_string):
        result_list = []
        db = get_db()
        results = db.execute(
            "SELECT * FROM ratings WHERE professor_name = ? OR course_code = ?", (query_string, query_string)
        ).fetchall()
        if not results:
            return None

        for rating in results:
            result = {
                'id':rating[0],
                'courseCode': rating[2],
                'courseName': rating[3],
                'name': rating[4],
                'ratings': [rating[5], rating[6], rating[7], rating[8]],
                'tags': [rating[9], rating[10], rating[11]],
                'review': rating[12],
                "flagCount": rating[13],
                "likeCount": rating[14],
                "dislikeCount": rating[15],
                "date": rating[16],
            }

            print("Rating: ", result)
            result_list.append(result)
        
        return result_list
    
    @staticmethod
    def get_autofill():
        result_list = []
        prof_list = []
        course_list = []
        db = get_db()
        results = db.execute(
            "SELECT * FROM ratings ORDER BY id DESC"
        ).fetchall()
        if not results:
            return None

        for rating in results:
            result_old = {
                'id':rating[0],
                'courseCode': rating[2],
                'courseName': rating[3],
                'name': rating[4],
                'ratings': [rating[5], rating[6], rating[7], rating[8]],
                'tags': [rating[9], rating[10], rating[11]],
                'review': rating[12]
            }

            # avoid duplicates and limit to less than 3 results
            if rating[4] not in prof_list:
                prof_list.append(rating[4])
            course_syntax = rating[3] + ' (' + rating[2] + ')'
            if course_syntax not in course_list:
                course_list.append(course_syntax)
        
        result_list = {
            'prof_list' : prof_list,
            'course_list' : course_list
        } 

        return result_list
    
    @staticmethod
    def update_count(id, count_type, count_update):
        db = get_db()

        # check if entry exists
        existing_entry = db.execute(
            "SELECT * FROM ratings WHERE id = ?",
            (id,)
        ).fetchone()

        if not existing_entry:
            return None

        if count_type == "flag_count":
            count_index = 13
        elif count_type == "like_count":
            count_index = 14
        elif count_type == "dislike_count":
            count_index = 15
        else:
            raise ValueError("Invalid count_type. Use 'flag_count', 'like_count', or 'dislike_count'.")

        # update specified count
        new_count = existing_entry[count_index] + count_update
        db.execute(
            f"UPDATE ratings SET {count_type} = ? WHERE id = ?",
            (new_count, id)
        )
        db.commit()

        return new_count

    @staticmethod
    def get_top_review():
        db = get_db()
        rating = db.execute(
            """
            SELECT * FROM ratings 
            WHERE (
                rating_prof + rate_cvn_quality + rate_cvn_resources + rate_cvn_difficulty
            ) / 4 >= 4
            ORDER BY RANDOM()
            LIMIT 1;
            """
        ).fetchone()
        if not rating:
            return None
    
        try:
            result = {
                'id':rating[0],
                'courseCode': rating[2],
                'courseName': rating[3],
                'name': rating[4],
                'ratings': [rating[5], rating[6], rating[7], rating[8]],
                'tags': [rating[9], rating[10], rating[11]],
                'review': rating[12]
            }
        except Exception as e:
            print("ERROR: ", str(e))
        
        return result