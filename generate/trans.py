# import os
# from google.cloud import translate_v2 as translate

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"E:\7th_sem\NLP\Project_working_area\django_implementation\text-generate-main\generate\hip-heading-283511-99e61f012135.json"
# from googletrans import Translator
# import googletrans
# def text_translate(txt,lang):
#     k=googletrans.LANGUAGES
#     lng2 = lang
#     for key, value in k.items():
#         if lng2 == value:
#             y=key



#     translate_client = translate.Client()

#     text = txt
#     target = y

#     output = translate_client.translate(
#         text,
#         target_language=target
#     )

#     return (output['translatedText'])

# print(text_translate('ఆమె ఆ తరగతిలో బాగానే ఉంది, నేను కోరుకున్నాను','english'))
a=46
if a%10!=0:
    print(a/10)
    a = ((a/10)+1)*10
    print(a)