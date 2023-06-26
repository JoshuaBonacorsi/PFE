import openai

def build_prompt(dates) :
    if dates != "" :
        prompt = f"""
    "Bonjour, j'ai travaillé sur un modèle d'apprentissage automatique qui avait pour tâche d'analyser l'ensemble de données NAB (Nyc_Taxi). Cet ensemble de données comprend des informations détaillées sur la demande de taxis sur une période significative à New York. Le modèle a signalé certaines dates où il y a eu des anomalies inattendues dans la demande de taxis - des situations où le nombre de courses de taxis demandées s'écartait significativement de la norme.

    Le but de cet exercice n'était pas seulement de détecter des anomalies, mais aussi de comprendre les raisons possibles qui les sous-tendent. Pour ce faire, nous devons approfondir ces dates, enquêter sur les événements potentiels en corrélation, les conditions météorologiques, les facteurs socio-économiques ou tout autre détail pertinent qui pourrait expliquer ces changements soudains de demande.

    Les dates que le modèle a signalées comme anormales sont :
    ```
    {dates}
    ```

    Pourriez-vous utiliser les données disponibles, ainsi que toute autre source de données externe à laquelle vous avez accès, pour m'aider à comprendre ce qui pourrait avoir causé ces anomalies dans la demande de taxis? Quels sont les facteurs potentiels qui auraient pu influencer une demande de taxis aussi inhabituelle à ces dates?

    Pour préciser davantage, vous pourriez considérer les questions suivantes :

    Y a-t-il eu des événements significatifs, tels que des concerts, des matchs de sport ou des festivals à ces dates qui auraient pu conduire à une augmentation de la demande de taxis ?
    Y a-t-il eu des conditions météorologiques inhabituelles à ces dates qui auraient pu provoquer une baisse de la demande de taxis ?
    Y a-t-il eu une perturbation des transports publics à ces dates qui aurait pu entraîner une augmentation de la demande de taxis ?
    Existe-t-il un facteur commun à ces dates qui pourrait suggérer une tendance plus large que le modèle n'a pas pris en compte ?
    Votre analyse approfondie et vos idées m'aideront à mieux comprendre ces anomalies et, plus important encore, à améliorer la précision et la fiabilité des prédictions du modèle à l'avenir.
    "
    """
    else :
        prompt = "Bonjour, je veux simplement que tu me salues et que tu me dises que tu es désolé car je ne t'ai pas fourni de dates à analyser !"
    return(prompt)

def get_completion(openai,prompt, model="gpt-4"):
    messages = [{"role": "system", "content": f"""
    Voici des éléments pouvant t'être utiles : 
    - Le jour précédant le marathon de New York : 2014-11-01
    - Thanksgiving : 2014-11-27
    - Période de Noël : 2014-12-24 à 2014-12-28
    - Jour de l'An : 2015-01-01
    - Tempête de neige : 2015-01-26 à 2015-01-27""",
                "role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

