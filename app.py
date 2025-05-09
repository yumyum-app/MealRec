import os
import pickle
import ast
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Global variables for model and data
DATA_PATH = "CCMR/data/recipe.csv"
MODEL_DIR = "models"
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
SVD_MODEL_PATH = os.path.join(MODEL_DIR, "svd_model.pkl")
LATENT_MATRIX_PATH = os.path.join(MODEL_DIR, "latent_matrix.pkl")
DATAFRAME_PATH = os.path.join(MODEL_DIR, "dataframe.pkl")

# Allowed options for the new input parameters
ALLOWED_TYPES = ["sweet", "spicy", "savory"]
ALLOWED_MEALTYPE = ["heavy", "not heavy"]
ALLOWED_DIETARY_NEEDS = ["healthy", "anything"]
ALLOWED_CUISINE_CATEGORY = ["mexican", "italian", "japanese", "indian", "american", "anything"]
ALLOWED_INGREDIENTS_RESTRICTION = ["red meat", "dairy", "nuts", "eggs", "fish", "no restrictions"]
ALLOWED_TIME = ["breakfast", "lunch", "dinner", "snack"]

# Helper function to evaluate nutritional health
def is_recipe_healthy(tags):
    tags = tags.lower()
    if "healthy" in tags or "low-calorie" in tags or "low-fat" in tags or "low-carb" in tags:
        return True
    return False

def is_recipe_heavy(tags):
    tags = tags.lower()
    if "heavy" in tags or "high-calorie" in tags or "high-fat" in tags or "high-carb" in tags:
        return True
    return False

def train_and_save_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")
    df = df.dropna()

    df['ingredients'] = df['ingredients'].apply(
        lambda x: " ".join([ingredient.strip() for ingredient in x.split('^') if ingredient])
    )
    df['tags'] = df['tags'].apply(
        lambda x: " ".join([tag.strip() for tag in x.split(';') if tag])
    )
    df['category'] = df['category'].str.lower()
    df['text_features'] = df['ingredients'] + " " + df['tags']

    # Create the is_healthy column (ensure it's always created)
    df['is_healthy'] = df['tags'].apply(is_recipe_healthy)
    df['is_heavy'] = df['tags'].apply(is_recipe_heavy)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df['text_features'])
    n_components = 100 if tfidf_matrix.shape[1] >= 100 else max(tfidf_matrix.shape[1] - 1, 1)
    svd_model = TruncatedSVD(n_components=n_components, random_state=42)
    latent_matrix = svd_model.fit_transform(tfidf_matrix)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(SVD_MODEL_PATH, "wb") as f:
        pickle.dump(svd_model, f)
    with open(LATENT_MATRIX_PATH, "wb") as f:
        pickle.dump(latent_matrix, f)
    with open(DATAFRAME_PATH, "wb") as f:
        pickle.dump(df, f)
    print("Models and data have been trained and saved.")
    return df, vectorizer, svd_model, latent_matrix

def load_models():
    try:
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        with open(SVD_MODEL_PATH, "rb") as f:
            svd_model = pickle.load(f)
        with open(LATENT_MATRIX_PATH, "rb") as f:
            latent_matrix = pickle.load(f)
        with open(DATAFRAME_PATH, "rb") as f:
            df = pickle.load(f)
        # In case the "is_healthy" column is missing, create it.
        if "is_healthy" not in df.columns:
            df["is_healthy"] = df["tags"].apply(is_recipe_healthy)
        print("Models and data loaded from disk.")
    except Exception as e:
        print("Error loading models, retraining:", e)
        df, vectorizer, svd_model, latent_matrix = train_and_save_models()
    return df, vectorizer, svd_model, latent_matrix

# Initialize models and data
df, vectorizer, svd_model, latent_matrix = load_models()

def get_best_recommendations(filtered_df, latent_matrix, top_n=3):
    recommendations = []
    if not filtered_df.empty:
        idx = filtered_df.index[0]
        target_vector = latent_matrix[idx].reshape(1, -1)
        sims = cosine_similarity(latent_matrix, target_vector).ravel()
        sim_indices = np.argsort(sims)[::-1]
        filtered_indices = [i for i in sim_indices if i in filtered_df.index]
        recommended = df.loc[filtered_indices].head(top_n)
        recommendations = recommended[["recipe_name", "recipe_id"]].to_dict('records')
    
    if len(recommendations) < top_n:
        additional_needed = top_n - len(recommendations)
        all_indices = df.index.tolist()
        filtered_indices = filtered_df.index.tolist() if not filtered_df.empty else []
        fallback_indices = [i for i in all_indices if i not in filtered_indices][:additional_needed]
        fallback_recommendations = df.loc[fallback_indices][["recipe_name", "recipe_id"]].to_dict('records')
        recommendations.extend(fallback_recommendations)
    
    return recommendations

@app.route('/predict', methods=["POST"])
def predict():
    """
    Input:
    {
       "data":{
          "food_id": "(1 buah id dari makanan yang disukai user)",
          "type": "spicy", // atau sweet, savory
          "mealtype": "heavy", // atau not heavy
          "dietary_needs": "healthy", // atau anything
          "cuisine_category": "mexican", // atau italian, japanese, indian, american, anything
          "ingredients-restriction": "red meat", // atau dairy, nuts, eggs, fish, no restrictions
          "time": "dinner" // atau breakfast, lunch, snack
       }
    }
    Output:
    {
       "status":200,
       "lists":[
          {
             "name": "Nasi Goreng",
             "id": 20
          },
          {
             "name": "Nasi Ayam",
             "id": 21
          }
       ]
    }
    """
    try:
        req_data = request.get_json()
        if not req_data or "data" not in req_data:
            return jsonify({"status": 400, "error": "Invalid request format."}), 400
        data = req_data["data"]

        # Extract parameters and normalize
        type_val = str(data.get("type", "")).strip().lower()
        mealtype_val = str(data.get("mealtype", "")).strip().lower()
        dietary_needs_val = str(data.get("dietary_needs", "")).strip().lower()
        cuisine_val = str(data.get("cuisine_category", "")).strip().lower()
        ingredients_rest_val = str(data.get("ingredients-restriction", "")).strip().lower()
        time_val = str(data.get("time", "")).strip().lower()

        # Validate each input against allowed options
        errors = {}
        if type_val not in ALLOWED_TYPES:
            errors["type"] = f"Allowed options: {ALLOWED_TYPES}"
        if mealtype_val not in ALLOWED_MEALTYPE:
            errors["mealtype"] = f"Allowed options: {ALLOWED_MEALTYPE}"
        if dietary_needs_val not in ALLOWED_DIETARY_NEEDS:
            errors["dietary_needs"] = f"Allowed options: {ALLOWED_DIETARY_NEEDS}"
        if cuisine_val not in ALLOWED_CUISINE_CATEGORY:
            errors["cuisine_category"] = f"Allowed options: {ALLOWED_CUISINE_CATEGORY}"
        if ingredients_rest_val not in ALLOWED_INGREDIENTS_RESTRICTION:
            errors["ingredients-restriction"] = f"Allowed options: {ALLOWED_INGREDIENTS_RESTRICTION}"
        if time_val not in ALLOWED_TIME:
            errors["time"] = f"Allowed options: {ALLOWED_TIME}"

        if errors:
            return jsonify({"status": 400, "error": "Invalid input parameters", "details": errors}), 400

        # Use complete dataframe for recommendations
        filtered_df = df.copy()

        # Rule for spicy: if type is spicy, search tags for chili-related keywords
        if type_val == "spicy":
            filtered_df = filtered_df[filtered_df['tags'].str.contains("chili|pepper|cayenne|jalapeno", case=False, na=False)]
        elif type_val == "sweet":
            filtered_df = filtered_df[filtered_df['tags'].str.contains("sweet", na=False)]
        elif type_val == "savory":
            filtered_df = filtered_df[filtered_df['tags'].str.contains("savory", na=False)]

        # Rule for healthy dietary needs: if healthy then only take recipes that satisfy our nutrition criteria
        if dietary_needs_val == "healthy":
            filtered_df = filtered_df[filtered_df["is_healthy"] == True]

        # Filter based on cuisine category (if not "anything")
        if cuisine_val and cuisine_val != "anything":
            filtered_df = filtered_df[filtered_df['tags'].str.contains(cuisine_val, na=False)]
        
        # Filter based on time (if not "anything")
        if time_val and time_val != "anything":
            filtered_df = filtered_df[filtered_df['tags'].str.contains(time_val, na=False)]
        
        # Filter based on ingredients restriction (if not "no restrictions")
        if ingredients_rest_val and ingredients_rest_val != "no restrictions":
            if ingredients_rest_val == "red meat":
                filtered_df = filtered_df[~filtered_df['tags'].str.contains("beef|buffalo|pork", case=False, na=False)]
            else:
                filtered_df = filtered_df[~filtered_df['ingredients'].str.contains(ingredients_rest_val, na=False)]

        recommendations = get_best_recommendations(filtered_df, latent_matrix, top_n=3)
        rec_list = []
        for rec in recommendations:
            rec_list.append({
                "name": rec["recipe_name"],
                "id": int(rec["recipe_id"])
            })
        return jsonify({"status": 200, "lists": rec_list})
    except Exception as e:
        return jsonify({"status": 500, "error": str(e)}), 500

@app.route('/similar', methods=["POST"])
def similar():
    """
    Input:
    {
       "data": {
           "food_id": 20 
           // or "food_id": [20, 25, 30] for multiple ids
       }
    }
    Output:
    {
       "status": 200,
       "results": [
           {
               "original_food_id": 20,
               "similar": [
                   { "name": "Nasi Goreng", "id": 20 },
                   { "name": "Nasi Ayam", "id": 21 },
                   { "name": "Mie Goreng", "id": 22 }
               ]
           },
           {
               "original_food_id": 25,
               "similar": [
                   { "name": "Sate Ayam", "id": 25 },
                   { "name": "Sate Kambing", "id": 26 },
                   { "name": "Soto Ayam", "id": 27 }
               ]
           }
           // More results for other food ids...
       ]
    }
    """
    try:
        req_data = request.get_json()
        if not req_data or "data" not in req_data or "food_id" not in req_data["data"]:
            return jsonify({"status": 400, "error": "Invalid request format."}), 400

        # Accept single food_id integer or a list of food_ids
        food_ids_input = req_data["data"]["food_id"]
        if not isinstance(food_ids_input, list):
            food_ids_input = [food_ids_input]

        results = []
        for food_id in food_ids_input:
            try:
                fid = int(food_id)
            except ValueError:
                continue  # Skip if not an integer

            matching_rows = df[df["recipe_id"] == fid]
            if matching_rows.empty:
                # You can either skip or add an entry with empty recommendations
                results.append({
                    "original_food_id": fid,
                    "similar": []
                })
                continue

            idx = matching_rows.index[0]
            target_vector = latent_matrix[idx].reshape(1, -1)
            sims = cosine_similarity(latent_matrix, target_vector).ravel()
            sim_indices = np.argsort(sims)[::-1]
            similar_recipes = []
            for i in sim_indices:
                if i == idx:
                    continue
                similar_recipes.append({
                    "name": df.loc[i]["recipe_name"],
                    "id": int(df.loc[i]["recipe_id"])
                })
                if len(similar_recipes) == 3:
                    break
            results.append({
                "original_food_id": fid,
                "similar": similar_recipes
            })

        return jsonify({"status": 200, "results": results})
    except Exception as e:
        return jsonify({"status": 500, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 