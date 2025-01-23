# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 18:25:42 2025

@author: Anna
"""

import csv


# function to create a lexicon dictionary #words_list, language_label, sentiment_score
def create_lexicon_dict():
    # Neutral sentiment words for each language
    neutral_bicol = ["tama", "kompleto", "masinup", "pasensya", "subong", "maligaya",
                     "simple", "hustisya", "sana", "matino", "mahusay lang", "higayon",
                     "wawari", "pasadong", "santuan", "seryo", "napag-usapan", "mga bagay",
                     "ordinaryo"]

    neutral_tagalog = ["tama", "normal", "hindi masama", "walang pagbabago", "maayos",
                       "simple", "katamtaman", "hustisya", "wala", "pantas", "magaan",
                       "tulungan", "pagkakataon", "tulungan", "nasa tamang landas",
                       "kalmado", "mga bagay", "nilalaman", "huwag kalimutan"]

    neutral_english = ["average", "neutral", "acceptable", "balanced", "normal",
                       "reasonable", "decent", "clear", "consistent", "fine", "calm",
                       "basic", "stable", "expected", "unremarkable", "moderate",
                       "mediocre", "regular", "reliable", "familiar", "standard",
                       "adequate", "common", "general", "typical", "ordinary", "fair",
                       "predictable", "unusual", "polite", "consistent"]

    # Load positive, negative words and stop words list
    positive_bicol = ["tultol", "maboot", "marhay", "magayon", "matibay", "maogmahon",
                      "madiskarte", "masinop", "marahay", "mapagkakatiwalaan", "mahalaga",
                      "inspirasyon", "pagtitiyaga", "dedikasyon", "kasanayan", "kaaram",
                      "matulungin", "marajaw", "masipag", "mahinahon", "maingat", "tapat",
                      "aktibo", "matakod", "puno nin enerhiya", "mapagpakumbaba",
                      "mapagpasalamat", "mapagkaogma", "puno nin paglaom", "maalalahanin",
                      "maunawain", "maparaan", "maka-Diyos", "maka-bayan", "maka-kalikasan",
                      "maserbisyo", "maka-komunidad", "masaliksik", "matiyaga", "mapagbigay",
                      "mapaglingkod"]

    positive_tagalog = ["mahusay", "magaling", "matagumpay", "matalino", "masikap",
                        "masinop", "malikhain", "mapagkakatiwalaan", "masunurin", "mahalaga",
                        "inspirasyon", "matiyaga", "matulungin", "mapagmalasakit",
                        "responsable", "masipag", "mahinahon", "maingat", "tapat",
                        "mapagbigay", "aktibo", "matapang", "puno ng enerhiya", "matatag",
                        "mapagpakumbaba", "mapagmahal", "puno ng pag-asa", "maalalahanin",
                        "maunawain", "maparaan", "maka-Diyos", "maka-bayan", "makakalikasan",
                        "makatao", "makabago", "makatarungan", "may disiplina",
                        "maserbisyo", "masaliksik", "mapaglingkod"]

    positive_english = ["excellent", "good", "successful", "intelligent", "diligent",
                        "efficient", "creative", "trustworthy", "obedient", "valuable",
                        "inspiration", "perseverance", "dedicated", "skillful", "wise",
                        "helpful", "compassionate", "responsible", "hardworking", "calm",
                        "careful", "honest", "generous", "active", "brave", "energetic",
                        "strong", "humble", "loving", "hopeful", "thoughtful", "loyal",
                        "understanding", "resourceful", "God-fearing", "patriotic",
                        "environmentally-conscious", "artistic", "scientific", "humane",
                        "charitable", "innovative", "just", "expert", "humanitarian",
                        "aesthetic", "scholarly", "spiritual", "technological", "logical",
                        "rational", "lawful", "eco-friendly", "creative", "strategic",
                        "devout", "altruistic", "nationalistic", "proficient", "ethical",
                        "designer", "culturally-aware", "economical", "mathematical",
                        "musical", "philosophical", "leadership", "research-oriented",
                        "community-focused", "communicative", "digital", "genius",
                        "patient", "comprehensive", "imaginative", "artful",
                        "environmental", "service-oriented", "up-to-date", "fair",
                        "unified", "health-conscious", "nature-lover", "analytical",
                        "thoughtful", "investigative", "excellence", "disciplined",
                        "literate", "intellectual", "experimental", "educator",
                        "learner-centric", "visionary", "team-player", "bureaucratic",
                        "charitable", "socially-aware", "patriotic", "skilled", "cultured",
                        "faithful", "pious", "eco-conscious", "crafty", "inventive",
                        "logical", "harmonious", "wise", "commanding", "exploring",
                        "structurally-aware", "societal", "interactive", "tech-savvy",
                        "ingenious", "steadfast", "exhaustive", "resourceful", "elegant",
                        "eco-friendly", "public-serving", "timely", "equitable",
                        "inclusive", "fitness-minded", "conservationist", "theoretical",
                        "reflective", "empirical", "excellence", "methodical", "scholarly",
                        "intellectual", "scientifically-minded", "teacher-like",
                        "student-focused", "visionary", "team-player", "administrative",
                        "charitable", "societal", "civic-minded", "skilled",
                        "knowledgeable"]

    negative_bicol = ["bakong marhay", "maaramon marhay", "dae magayon", "madunongon",
                      "dae maboot", "bakog marajaw", "dae mapagkakatiwalaan",
                      "dae masinop", "maluya", "dae maalalahanin", "dae maunawain",
                      "dae maparaan", "dae maka-Diyos", "dae maka-bayan",
                      "dae maka-kalikasan", "dae makatao", "dae makabago",
                      "dae makatarungan", "maisip", "mababa", "tamad", "mahina",
                      "masama", "mapanganib", "masungit", "makasarili", "mapagmataas",
                      "matigason an payo", "mapanghusga", "mapagsarili", "mayong pakiaram",
                      "mayong tiwala", "mayong kwenta", "mapurol an payo", "taksil",
                      "iresponsable", "makupad", "masalimuot", "magulo", "pangit",
                      "magaspang an ugali", "bastos", "mapagpanggap", "sakim", "matakaw",
                      "mapang-api", "maramot", "balasubas", "malupit", "mapanlinlang",
                      "mapanira", "maingay", "abusado", "mapang-abuso", "mapanghamak",
                      "mapangwasak", "mapag-imbot", "mapusok", "mapagsamantala",
                      "mapang-insulto", "mapang-abala", "mapangmaliit", "daeng modo",
                      "daeng malasakit", "mayong respeto", "mayong utang na loob",
                      "mayong pagpapahalaga", "mayong gana", "mayong puso",
                      "mayong kaluluwa", "mayong dignidad", "mayong prinsipyo",
                      "mayong moralidad", "mayong pananampalataya", "mayong paniniwala",
                      "mayong pag-asa", "bakong patas", "mayong direksyon",
                      "mayong kakayahan", "mayong pananagutan", "walang pagkilala",
                      "mayong aram", "mayong serbi", "mayong pagka-unawa", "mayong boses",
                      "mayong paninindugan", "mayong mapapala", "ignorante",
                      "mayong maginibo", "dae maboot", "nagduduwa-duwa"]

    negative_tagalog = ["mababa", "tamad", "mahina", "masama", "mapanganib", "masungit",
                        "makasarili", "mapagmataas", "matigas ang ulo", "mapanghusga",
                        "mapagsarili", "walang pakialam", "walang tiwala", "walang galang",
                        "walang kwenta", "mahina ang ulo", "taksil", "iresponsable",
                        "makupad", "masalimuot", "magulo", "pangit", "magaspang", "bastos",
                        "mapagpanggap", "masakim", "matakaw", "mapang-api", "maramot",
                        "balasubas", "malupit", "masungit", "mapagbiro", "mapanlinlang",
                        "mapagmataas", "mapanira", "maingay", "walang silbi", "walang alam",
                        "mapang-abuso", "mapanghamak", "mapangwasak", "mapag-imbot",
                        "mapang-api", "mapusok", "malupit", "mapagpanggap", "masungit",
                        "mapagsamantala", "mapang-insulto", "mapagmataas", "mapang-abala",
                        "mapangmaliit", "walang modo", "walang galang", "walang malasakit",
                        "walang respeto", "walang utang na loob", "walang pagpapahalaga",
                        "walang kasiglahaan", "walang gana", "walang malasakit",
                        "walang puso", "walang kaluluwa", "walang dignidad",
                        "walang prinsipyo", "walang moralidad",
                        "walang pananampalataya", "walang paniniwala",
                        "walang pag-asa", "walang katarungan", "walang batas",
                        "walang direksyon", "walang kapayapaan", "walang kabuluhan",
                        "walang pakialam", "walang patutunguhan", "walang silbi",
                        "walang kakayahan", "walang layunin", "walang disiplina",
                        "walang paki", "walang malay", "walang saysay", "walang alaala",
                        "walang pag-iisip", "walang kaligayahan", "walang kasiyahan",
                        "walang sigla", "walang lakas", "walang tiyaga", "walang malasakit",
                        "walang tibay", "walang tapang", "walang pagpapahalaga",
                        "walang pananagutan", "walang pagkilala", "walang kaalaman",
                        "walang kasaysayan", "walang pag-unawa", "walang kakayahan",
                        "walang boses", "walang paninindigan", "walang mapapala",
                        "walang kinikilingan", "walang malay", "walang nalalaman",
                        "walang magawa", "walang tiwala", "walang malasakit",
                        "walang pananagutan", "walang kabaitan", "walang respeto",
                        "walang tiwala", "walang paninindigan", "walang hinaharap",
                        "walang paki", "walang patutunguhan", "walang layunin",
                        "walang saysay", "walang sigla", "walang patutunguhan",
                        "walang lakas", "walang tiyaga", "walang malasakit",
                        "walang tibay", "walang tapang", "walang pagpapahalaga",
                        "walang pananagutan", "walang kinikilingan", "walang layunin",
                        "walang kalaban-laban", "walang saysay", "walang kinikilingan",
                        "walang patutunguhan", "walang kakayahan", "walang pangarap",
                        "walang patutunguhan", "walang saysay", "walang kinikilingan",
                        "walang patutunguhan", "walang kinikilingan", "walang malay",
                        "walang saysay", "walang patutunguhan", "walang silbi",
                        "walang kinikilingan", "walang saysay", "walang malay",
                        "walang patutunguhan"]

    negative_english = ["low", "lazy", "weak", "bad", "dangerous", "grouchy", "selfish",
                        "arrogant", "stubborn", "judgemental", "self-centered",
                        "indifferent", "untrustworthy", "disrespectful", "worthless",
                        "dumb", "traitorous", "irresponsible", "slow", "complicated",
                        "chaotic", "ugly", "rude", "impolite", "pretentious", "greedy",
                        "gluttonous", "oppressive", "stingy", "deceitful", "cruel",
                        "sarcastic", "deceptive", "egotistical", "destructive", "noisy",
                        "useless", "ignorant", "abusive", "contemptuous", "destructive",
                        "envious", "oppressive", "impulsive", "harsh", "hypocritical",
                        "grumpy", "exploitative", "insulting", "overbearing", "disruptive",
                        "belittling", "discourteous", "impolite", "uncaring",
                        "disrespectful", "ungrateful", "unappreciative", "lethargic",
                        "unmotivated", "insensitive", "heartless", "soulless",
                        "undignified", "unprincipled", "immoral", "faithless",
                        "unbelieving", "hopeless", "unjust", "lawless", "directionless",
                        "unpeaceful", "pointless", "unconcerned", "aimless", "ineffectual",
                        "incompetent", "purposeless", "undisciplined", "apathetic",
                        "unconscious", "meaningless", "forgetful", "thoughtless", "joyless",
                        "unsatisfied", "spiritless", "powerless", "impatient", "uncaring",
                        "unreliable", "cowardly", "unappreciative", "unaccountable",
                        "unrecognized", "uninformed", "worldly", "unhistoric",
                        "uncomprehending", "unable", "voiceless", "unstable", "unrewarding",
                        "unbiased", "innocent", "ignorant", "helpless", "distrusting",
                        "indifferent", "irresponsible", "unkind", "disrespectful",
                        "suspicious", "unstable", "futureless", "apathetic", "unaimed",
                        "aimless", "insignificant", "lifeless", "aimless", "weak",
                        "impatient", "uncaring", "unstable", "cowardly", "inconsiderate",
                        "careless", "indifferent", "unmotivated", "helpless", "useless",
                        "unbiased", "uncertain", "inept", "dreamless", "lost", "worthless",
                        "neutral", "disoriented", "uninspired", "oblivious", "trivial",
                        "misguided", "ineffective", "apathetic", "unimportant", "unaware",
                        "vague"]

    # load common stop words
    stop_words_bicol = ["ako", "ko", "sa", "sinda", "kami", "kita", "mo", "iyo", "saiya",
                        "siya", "igwa", "kaining", "si", "an", "na", "dahil", "kung",
                        "maski", "habang", "giraray", "dapat", "na", "pa", "daw", "ngunyan",
                        "bago", "pagkatapos", "kaparteng", "para", "kan", "as", "ba",
                        "dahil", "maski", "alagad", "sa", "ngani", "igwa", "saro", "duwa",
                        "piga", "ini", "ito", "pwedeng", "ngani", "pa", "nga", "puwedeng",
                        "siguro", "labi", "aramon", "ngaya"]

    stop_words_tagalog = ["ako", "ko", "akin", "kami", "tayo", "namin", "natin", "ka",
                          "ikaw", "mo", "iyo", "kayo", "ninyo", "siya", "niya", "kaniya",
                          "sila", "nila", "kanila", "ano", "alin", "sino", "nito", "dito",
                          "iyan", "iyon", "am", "ay", "si", "ang", "at", "ng", "o", "dahil",
                          "kapag", "habang", "mula", "hanggang", "sa", "mula", "pa", "na",
                          "ngayon", "bago", "pagkatapos", "kasama", "para", "ng", "sa",
                          "tulad", "hindi", "wala", "kaya", "lamang", "rin", "din", "ito",
                          "iyan", "iyon", "muli", "pang", "dito", "doon", "saan", "bakit",
                          "paano", "lahat", "bawat", "ilan", "mas", "karamihan", "iba",
                          "kaunti", "ilan", "mga", "ganito", "huwag", "hindi", "walang",
                          "mismo", "iyan", "kaya", "masyado", "nito", "t", "ay", "maaari",
                          "magiging", "lamang", "huwag", "dapat", "ngayon"]

    stop_words_english = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves",
                          "you", "your", "yours", "yourself", "yourselves", "he", "him",
                          "his", "himself", "she", "her", "hers", "herself", "it", "its",
                          "itself", "they", "them", "their", "theirs", "themselves", "what",
                          "which", "who", "whom", "this", "that", "these", "those", "am",
                          "is", "are", "was", "were", "be", "been", "being", "have", "has",
                          "had", "having", "do", "does", "did", "doing", "a", "an", "the",
                          "and", "but", "if", "or", "because", "as", "until", "while", "of",
                          "at", "by", "for", "with", "about", "against", "between", "into",
                          "through", "during", "before", "after", "above", "below", "to",
                          "from", "up", "down", "in", "out", "on", "off", "over", "under",
                          "again", "further", "then", "once", "here", "there", "when",
                          "where", "why", "how", "all", "any", "both", "each", "few",
                          "more", "most", "other", "some", "such", "no", "nor", "not",
                          "only", "own", "same", "so", "than", "too", "very", "s", "t",
                          "can", "will", "just", "don", "should", "now"]

    lexicon_dict = {}

    # Populate lexicon dictionary with neutral words
    for word in neutral_bicol:
        lexicon_dict[word] = {"language": "bk", "sentiment_score": 0}

    for word in neutral_tagalog:
        lexicon_dict[word] = {"language": "tl", "sentiment_score": 0}

    for word in neutral_english:
        lexicon_dict[word] = {"language": "en", "sentiment_score": 0}

    # Assign sentiment scores and language labels to positive Bicol words
    for word in positive_bicol:
        lexicon_dict[word] = {"sentiment_score": 1, "language": "bk"}

    # Assign sentiment scores and language labels to positive Bicol words
    for word in positive_tagalog:
        lexicon_dict[word] = {"sentiment_score": 1, "language": "tl"}

    # Assign sentiment scores and language labels to positive Bicol words
    for word in positive_english:
        lexicon_dict[word] = {"sentiment_score": 1, "language": "en"}

    # Assign sentiment scores and language labels to negative Bicol words
    for word in negative_bicol:
        lexicon_dict[word] = {"sentiment_score": -1, "language": "bk"}

    # Assign sentiment scores and language labels to negative Tagalog words
    for word in negative_tagalog:
        lexicon_dict[word] = {"sentiment_score": -1, "language": "tl"}

    # Assign sentiment scores and language labels to negative English words
    for word in negative_english:
        lexicon_dict[word] = {"sentiment_score": -1, "language": "en"}

    # Assign sentiment scores and language labels to stop words
    for word in stop_words_bicol:
        lexicon_dict[word] = {"sentiment_score": 0, "language": "bk"}

    for word in stop_words_tagalog:
        lexicon_dict[word] = {"sentiment_score": 0, "language": "tl"}

    for word in stop_words_english:
        lexicon_dict[word] = {"sentiment_score": 0, "language": "en"}

    # Write the lexicon to CSV
    with open('lex.csv', 'w', newline='') as csvfile:
        fieldnames = ['word', 'language', 'sentiment_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for word, data in lexicon_dict.items():
            writer.writerow({'word': word, 'language': data['language'], 'sentiment_score': data['sentiment_score']})


# Example usage
create_lexicon_dict()