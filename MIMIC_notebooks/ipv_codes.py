NHAS_IPV_CODES = ['E9601', 'E9673', '99580', '99581', '99582', '99583', '99585', 'V1541', 'V1542', 'V6110', 'V6111']
# National Hospital Ambulatory  Survey 2020 
# ICD-9 codes used: codes found in the cause of injury (rape [E960.1], spouse abuse [E967.3]) and 
# diagnosis (adult abuse [995.80–995.83, 995.85], history of violence [V15.41, V15.42], 
# and counseling [V61.10, V61.11]) fields

primary_codes = ['E9673', '99581', '99580']
secondary_codes = ['E9673', 'E9600',  '99581',  'E9680', 'E9681','E9682', 'E9683','E9684', 'E9685','E9686', 'E9687', 'E9688', 'E9689',  '99583', 
'V6110', 'E9620', 'E966', 'E9679', 'E9671', 'V6111',  '99585','E961']
OREGON_IPV_CODES = list(set(primary_codes + secondary_codes))
# Oregon ED code paper 2008
# Primary ICD-9 codes used:  E967.3 (battering by intimate partner), 
#                            995.81 (adult physical abuse), 995.80 (adult maltreatment)
# Secondary ICD-9 codes used: E967.3, E960.0 (unarmed fight/brawl), 995.81 (adult physical abuse),  
# E968.0-E968.9 (assault by other unspecified means), 995.83 (Adult sexual abuse), 
# V61.10 (marital/partner  counseling), E962.0-E966 (assault: poison, hanging, etc), E967.9 (battering by unspecified person), E967.1 (battering by  other specified person), 995.85 adult abuse/neglect
# V61.11 (IPV counseling), E961.0 assault by corrosive caustic substance
    
USED_IPV_CODES = ['E9673']
# United States ED survey 2014
# ICD-9 codes used:  E967.3 (battering by intimate partner)
# Estimated p(y) = .02% 


ICD10_IPV_CODES = ['T7421XD', 'T7491XA', 'T7421XS', 'T7411XS', 'T7421XA', 'T7431XA', 'T7411XA',
                   'T7621XA', 'T7691XA', 'T7601XA', 'T7631XA', 'T7611XA', 'T7621XS', 'T7601XD']

CODE_SETS = [('NHAS Study', NHAS_IPV_CODES), ('Oregon Study', OREGON_IPV_CODES), ('US ED Study', USED_IPV_CODES),
             ('ICD10 Codes', ICD10_IPV_CODES)]
KW_SETS = [['adult physical abuse', 'adult abuse'], 
                ['adult physical abuse', 'adult abuse', 'assault'],
                ['adult physical abuse', 'adult abuse', 'maltreatment'],
          ['partner'],
          ['abuse']]

IPV_RELATED_CODES = list(set(NHAS_IPV_CODES + OREGON_IPV_CODES + USED_IPV_CODES + ICD10_IPV_CODES))
IPV_RELATED_KWS = list(set([kw for kw_set in KW_SETS for kw in kw_set]))

T74_CODES = ['T74']
T76_CODES = ['T76']
Y_CODES = ['Y070', 'Y071', 'Y074']

# E9673 = Perpetrator of child and adult abuse, by spouse or partner
# E9671 = perpetrator of child/adult abuse, by unspecified person
# E9670 = perpetrator of child/adult abuse, by father stepfather etc.
# V6111 = Counseling for partner abuse
GOLD_STANDARD_IPV_CODES_1 = ['E9673', 'E9671', 'E9670','V6111']

# Adding in Adult physical abuse related codes
# 99581 =  Adult physical abuse
# T7411XA = Adult physical abuse, confirmed, initial encounter
# T7411XD = Adult physical abuse, confirmed, subsequent
# T7411XS = Adult physical abuse, confirmed, sequela

GOLD_STANDARD_IPV_CODES_2 = ['E9673', 'E9671', 'E9670','V6111',
                             '99581', 'T7411XA', 'T7411XD', 'T7411XS']

# Adding in Other adult maltreatment (unspecified). 'Maltreatment' encompasses physical, sexual, and emotional abuse.
# We exclude sexual and emotional abuse.

GOLD_STANDARD_IPV_CODES_3 = ['E9673', 'E9671', 'E9670','V6111', 
                             '99581', 'T7411XA', 'T7411XD', 'T7411XS', 
                             'T7491XA', 'T7491XD', 'T7491XS']
# Adding in suspected codes
# T7611XA = Adult physical abuse, suspected, initial encounter
GOLD_STANDARD_IPV_CODES_4 = ['E9673', 'E9671', 'E9670','V6111', 
                             '99581', 'T7411XA', 'T7411XD', 'T7411XS', # Adult physical abuse:  confirmed
                             'T7491XA', 'T7491XD', 'T7491XS', # Adult maltreatment (other): confirmed
                             'T7611', 'T7611XA', 'T7611XD', 'T7611XS', # Adult physical abuse: suspected
                             'T7692', 'T7692XA', 'T7692XD', 'T7692XS']  # Adult maltreatment (other): suspected 


SUSPICIOUS_SYMPTOMS_ICD9_CODE_PREFIXES = ['95901', '7842', '9108', '9100', '920']
SUSPICIOUS_SYMPTOMS_ICD10_CODE_PREFIXES = ['E963', '9947', # Assault by strangulation, asphyxiation  or strangulation
                                   'S100XX', 'S1080X', 'S1081X', 'S1090X', 'S1091X', 'S1093X', # Neck injuries
                                   'S0000X', 'S0001X', 'S0003X', 'S0990X', 'S0991X', 'S0993X', # Head injuries
                                   'S0010X', 'S0011X', 'S0012X', 'S00211', 'S00212', 'S00219', # Eye injuries
                                   'S00401', 'S00402', 'S00403', 'S00411', 'S00412', 'S00413', # Ear injuries
                                   'S00431', 'S00432', 'S00433', # Ear injuries
                                   'S00511', 'S00512', 'S00531', 'S00533'] # Lips / Oral Cavity Injuries

SUSPICIOUS_SYMPTOMS_ICD_CODES_PREFIXES = SUSPICIOUS_SYMPTOMS_ICD9_CODE_PREFIXES + SUSPICIOUS_SYMPTOMS_ICD10_CODE_PREFIXES

SUSPICIOUS_SYMPTOMS_ICD_CODES = ["95901", "7842", "9108", "9100", "920", "E963", "9947", "S100XXA", "S100XXD", "S100XXS", "S1080XA", "S1080XD", "S1080XS", "S1081XA", "S1081XD", "S1081XS", "S1090XA", "S1090XD", "S1090XS", "S1091XA", "S1091XD", "S1091XS", "S1093XA", "S1093XD", "S1093XS", "S0000XA", "S0000XD", "S0000XS", "S0001XA", "S0001XD", "S0001XS", "S0003XA", "S0003XD", "S0003XS", "S0990XA", "S0990XD", "S0990XS", "S0991XA", "S0991XD", "S0991XS", "S0993XA", "S0993XD", "S0993XS", "S0010XA", "S0010XD", "S0010XS", "S0011XA", "S0011XD", "S0011XS", "S0012XA", "S0012XD", "S0012XS", "S00211", "S00211A", "S00211D", "S00211S", "S00212", "S00212A", "S00212D", "S00212S", "S00219", "S00219A", "S00219D", "S00219S", "S00401", "S00401A", "S00401D", "S00401S", "S00402", "S00402A", "S00402D", "S00402S", "S00411", "S00411A", "S00411D", "S00411S", "S00412", "S00412A", "S00412D", "S00412S", "S00431", "S00431A", "S00431D", "S00431S", "S00432", "S00432A", "S00432D", "S00432S", "S00511", "S00511A", "S00511D", "S00511S", "S00512", "S00512A", "S00512D", "S00512S", "S00531", "S00531A", "S00531D", "S00531S"]

