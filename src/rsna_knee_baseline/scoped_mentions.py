"""#RSNA #Kaggle #Pesquisa — conservative L01 diagnostic, not clinical labels.

Keeps the legacy target vocabulary. New rules change assertion scope, not the
production teacher. Offsets refer to normalized text, not the original report.
Unsupported constructions abstain; no accuracy/confidence is inferred.
"""
from __future__ import annotations

import re
import unicodedata
from .lexicon import LEXICON, compile_patterns

RULE_VERSION = 'l01-scoped-mentions-v1'
STATES = ('present', 'absent', 'uncertain', 'unresolved', 'conflict', 'not_mentioned', 'excluded_only')
PATTERNS = {target: compile_patterns(terms) for target, terms in LEXICON.items()}
NEGATIVE = {
    'en': r'no|not|without|absent|intact|normal|preserved|unremarkable|negative for',
    'pt': r'nao|sem|ausente|integro|integra|integros|integras|preservado|preservada|normal',
    'es': r'no|sin|ausente|integro|integra|conservado|conservada|normal',
    'fr': r'pas de|aucun|aucune|sans|absent|absente|intact|normale|normal|preserve',
    'de': r'kein|keine|keinen|ohne|intakt|unauffallig',
    'nl': r'geen|zonder|intact|intacte|normaal|normale',
}
UNCERTAIN = {
    'en': r'possible|possibly|suspected|suspect|questionable|cannot exclude|cannot rule out|not excluded|not ruled out|may represent|could represent|no definite|no convincing|rule out',
    'pt': r'possivel|suspeita|suspeito|nao se pode excluir|nao exclui',
    'es': r'posible|sospecha|no se puede excluir',
    'fr': r'possible|suspect|suspecte|ne peut exclure',
    'de': r'verdacht|moglich|nicht ausgeschlossen',
    'nl': r'mogelijk|verdenking|niet uitgesloten',
}
EXCLUDED_HEADERS = ('clinical history','history','indication','clinical information',
    'comparison','indicacao','historia clinica','indicacion','antecedentes',
    'renseignements cliniques','klinische inlichtingen','diagnostische vraagstelling')
INCLUDED_HEADERS = ('findings','impression','conclusion','conclusions','technique',
    'achados','conclusao','tecnica','hallazgos','conclusiones','resultats','conclusie',
    'bevindingen','befund','beurteilung')
HEADERS = re.compile(r'(?<!\w)('+ '|'.join(map(re.escape, EXCLUDED_HEADERS+INCLUDED_HEADERS)) +r')\s*(?::|\n)')
BOUNDARY = re.compile(r'[.;!\n]+|\b(?:but|however|whereas|mas|porem|pero|mais|jedoch|aber|echter|maar)\b')
STRUCTURAL = {'ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA'}
ABNORMAL = re.compile(r'\b(?:tear|torn|rupture|ruptura|ruptured|sprain|lesion|lesao|dechirure|'
    r'osteoarthritis|osteoarthrosis|arthrosis|arthrose|artrose|artrosis|chondromalacia|condromalacia|'
    r'cartilage loss|cartilage defect|cartilage defects|chondral defect|chondral defects)\b')


def normalized(text):
    text = unicodedata.normalize('NFKD',str(text or '')).encode('ascii','ignore').decode('ascii').lower()
    return re.sub(r'[^\S\n]+',' ',text)


def matches(text, expressions):
    return [lang for lang, expression in expressions.items()
            if re.search(r'(?<!\w)(?:'+expression+r')(?!\w)',text)]


def sections(text):
    """Header scope ends at the next explicit recognized header, never silently."""
    headers = list(HEADERS.finditer(text))
    if not headers:
        yield 'unheaded',False,0,text
        return
    if headers[0].start(): yield 'unheaded',False,0,text[:headers[0].start()]
    for i,h in enumerate(headers):
        stop=headers[i+1].start() if i+1<len(headers) else len(text)
        yield h[1],h[1] in EXCLUDED_HEADERS,h.end(),text[h.end():stop]


def classify_clause(clause, target, distinct_targets):
    uncertain=matches(clause,UNCERTAIN)
    if '?' in clause or uncertain:
        return 'uncertain',uncertain,'question_or_uncertainty'
    # A shared clause can attach different predicates to different structures.
    # Do not guess dependency scope with a bag of cues.
    if len(distinct_targets)>1:
        return 'unresolved',[],'multiple_targets_in_clause'
    negative=matches(clause,NEGATIVE)
    if negative:
        # Negation plus an adversative/degree qualifier can change meaning.
        if re.search(r'\b(?:although|despite|except|otherwise|not only|no longer|ohne sichere|partially intact)\b',clause):
            return 'unresolved',negative,'complex_negation'
        return 'absent',negative,'explicit_negative_or_normal'
    if target in STRUCTURAL and not ABNORMAL.search(clause):
        return 'unresolved',[],'anatomy_without_supported_abnormality'
    return 'present',[],'asserted_finding'


def aggregate(states, excluded_count):
    unique=set(states)
    if 'present' in unique and 'absent' in unique: return 'conflict'
    if 'uncertain' in unique: return 'uncertain'
    if 'unresolved' in unique: return 'unresolved'
    if unique: return next(iter(unique))
    return 'excluded_only' if excluded_count else 'not_mentioned'


def analyze(text):
    result={t:{'state':'not_mentioned','evidence':[]} for t in LEXICON}
    for section,excluded,offset,body in sections(normalized(text)):
        start=0
        boundaries=list(BOUNDARY.finditer(body))
        spans=[(b.start(),b.end()) for b in boundaries]+[(len(body),len(body))]
        for stop,next_start in spans:
            clause=body[start:stop]
            found={t:[m for p in patterns for m in p.finditer(clause)] for t,patterns in PATTERNS.items()}
            targets=[t for t,ms in found.items() if ms]
            for t in targets:
                state,langs,reason=('excluded',[],'clinical_context_section') if excluded else classify_clause(clause,t,targets)
                for m in found[t]:
                    result[t]['evidence'].append({'state':state,'reason':reason,'section':section,
                        'normalized_span':[offset+start+m.start(),offset+start+m.end()],
                        'matched_term':m.group(),'clause':clause.strip(),'cue_languages':langs})
            start=next_start
    for row in result.values():
        states=[e['state'] for e in row['evidence'] if e['state']!='excluded']
        row['state']=aggregate(states,sum(e['state']=='excluded' for e in row['evidence']))
    return result
