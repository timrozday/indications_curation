import sys
import re
import json
import pickle
import rdflib
import itertools as it
from lxml import etree
from html import unescape
import copy
import sqlite3
from sqlite3 import IntegrityError

import pandas as pd
from tabulate import tabulate

import scispacy
import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES

from text_query import text_query as tq
from group_matches import group_matches as gm

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from matplotlib import pyplot as plt

import pprint
pp = pprint.PrettyPrinter(width=160, compact=False)
from datetime import datetime

import configparser

def init():
	set_working_dir("/Users/timrozday/Documents/manual_curation_dm_inds/csv_curation_answers")
	read_config()
	connect_dbs()
	load_indexes()
	r = gen_filter_rules_indexes()
	print_env()
	r = list_spls()

def connect_dbs():
    global indi_conn
    global thes_conn
    global index_conn
    global ca_conn

    indi_conn = sqlite3.connect("/Users/timrozday/Documents/dm_indi_sentences.sqlite")
    thes_conn = sqlite3.connect("/Users/timrozday/Documents/thesaurus_kmer_index.sqlite")
    index_conn = sqlite3.connect("/Users/timrozday/Documents/text_query_kmer_index.sqlite")
    ca_conn = sqlite3.connect("/Users/timrozday/Documents/manual_curation_dm_inds/csv_curation_answers/curation_answers.sqlite")

def close_dbs():
    try: ca_conn.close()
    except: pass
    try: indi_conn.close()
    except: pass
    try: thes_conn.close()
    except: pass
    try: index_conn.close()
    except: pass

def load_indexes():
    global equivalent_entities_groups_index
    global equivalent_entities_groups_index_r
    global disease_hierarchy_distance_index
    global disease_hierarchy_index
    global rev_disease_hierarchy_distance_index
    global node_index
    global node_index_r
    global source_index
    global source_index_r
    global rel_type_index
    global rel_type_index_r
    global zip_metadata
    global word_index
    global partial_word_index
    global s_len_index
    global id_onto_index
    global onto_index
    global onto_index_r
    global partial_code_index
    global code_name_index
    global nlp
    global lemmatizer

    nlp = spacy.load('en_core_sci_md')
    lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)

    with open("/Users/timrozday/Documents/vocab_harmonisation/equivalent_entities_groups_index.json",'rt') as f: equivalent_entities_groups_index = json.load(f)

    equivalent_entities_groups_index_r = {}
    for k,vs in equivalent_entities_groups_index.items():
        for v in vs:
            try: equivalent_entities_groups_index_r[v].add(k)
            except: equivalent_entities_groups_index_r[v] = {k}

    with open('/Users/timrozday/Documents/vocab_harmonisation/disease_hierarchy_index.json','rt') as f: disease_hierarchy_distance_index = json.load(f)

    disease_hierarchy_index = {}
    for source,hier_index in disease_hierarchy_distance_index.items():
        disease_hierarchy_index[source] = {}
        for k,vs in hier_index.items():
            disease_hierarchy_index[source][k] = {v for v,d in vs.items()}

    rev_disease_hierarchy_distance_index = {}
    for source,hier_index in disease_hierarchy_distance_index.items():
        rev_disease_hierarchy_distance_index[source] = {}
        for k,vs in hier_index.items():
            for v,d in vs.items():
                try: rev_disease_hierarchy_distance_index[v] = join_entities_dict({k:d}, rev_disease_hierarchy_distance_index[v])
                except: rev_disease_hierarchy_distance_index[v] = {k:d}

    with open("/Users/timrozday/Documents/vocab_harmonisation/node_index.pkl",'rb') as f: node_index = pickle.load(f)
    with open("/Users/timrozday/Documents/vocab_harmonisation/source_index.pkl",'rb') as f: source_index = pickle.load(f)
    with open("/Users/timrozday/Documents/vocab_harmonisation/rel_type_index.pkl",'rb') as f: rel_type_index = pickle.load(f)

    node_index_r = {}
    for k,v in node_index.items():
        node_index_r[v] = k

    source_index_r = {}
    for k,v in source_index.items():
        source_index_r[v] = k

    rel_type_index_r = {}
    for k,v in rel_type_index.items():
        rel_type_index_r[v] = k
    
    zip_metadata = json.load(open("/Users/timrozday/Documents/DailyMed data/zip_metadata.json",'rt'))

    with open('/Users/timrozday/Documents/onto_index/word_index.pkl', 'rb') as f: word_index = pickle.load(f)
    with open('/Users/timrozday/Documents/onto_index/partial_word_index.pkl', 'rb') as f: partial_word_index = pickle.load(f)
    with open('/Users/timrozday/Documents/onto_index/s_len_index.pkl', 'rb') as f: s_len_index = pickle.load(f)
    with open('/Users/timrozday/Documents/onto_index/id_onto_index.pkl', 'rb') as f: id_onto_index = pickle.load(f)
    with open('/Users/timrozday/Documents/onto_index/onto_index.pkl', 'rb') as f: onto_index = pickle.load(f)
    with open('/Users/timrozday/Documents/onto_index/onto_index_r.pkl', 'rb') as f: onto_index_r = pickle.load(f)

    with open('/Users/timrozday/Documents/onto_index/partial_code_index.pkl', 'rb') as f: partial_code_index = pickle.load(f)
    with open('/Users/timrozday/Documents/onto_index/code_name_index.pkl', 'rb') as f: code_name_index = pickle.load(f)

def setup_database(fn="/Users/timrozday/Documents/manual_curation_dm_inds/csv_curation_answers/curation_answers.sqlite"):
    ca_conn = sqlite3.connect(fn)

    ca_conn.execute("create table          spl(id INTEGER, set_id TEXT, date TEXT, title TEXT, version INTEGER, primary key(id))")
    ca_conn.execute("create table        nodes(id INTEGER, spl_id INTEGER, parent_node_id INTEGER, tag TEXT, loc TEXT, primary key(id), foreign key(parent_node_id) references nodes(id), foreign key(spl_id) references spl(id))")
    ca_conn.execute("create table    sentences(id INTEGER, parent_node_id INTEGER, spl_id INTEGER, loc TEXT, string TEXT, sentence TEXT, expanded_sentence TEXT, primary key(id), foreign key(parent_node_id) references nodes(id), foreign key(spl_id) references spl(id))")

    ca_conn.execute("create table        codes(id INTEGER, code TEXT, source TEXT, name TEXT, primary key(id))")
    ca_conn.execute("create table      answers(id INTEGER, answer_id INTEGER, group_id INTEGER, sentence_id INTEGER, locs TEXT, code TEXT, predicate_type TEXT, code_id INTEGER, true_match BOOLEAN, negative BOOLEAN, indication TEXT, never_match BOOLEAN, dont_match BOOLEAN, acronym BOOLEAN, note TEXT, timestamp TEXT, author TEXT, primary key(id), foreign key(sentence_id) references sentences(id), foreign key(code_id) references codes(id))")
    ca_conn.execute("create table  answers_hier(id INTEGER, child_id INTEGER, parent_id INTEGER, primary key(id), foreign key(child_id) references matches(id), foreign key(parent_id) references matches(id))")

    ca_conn.execute("create table  never_match(id INTEGER, spl_id INTEGER, code_id INTEGER, timestamp TEXT, author TEXT, primary key(id), foreign key(code_id) references codes(id), foreign key(spl_id) references spl(id))")
    ca_conn.execute("create table   dont_match(id INTEGER, spl_id INTEGER, code_id INTEGER, words TEXT, timestamp TEXT, author TEXT, primary key(id), foreign key(code_id) references codes(id), foreign key(spl_id) references spl(id))")
    ca_conn.execute("create table      acronym(id INTEGER, spl_id INTEGER, words TEXT, timestamp TEXT, author TEXT, primary key(id), foreign key(spl_id) references spl(id))")

    ca_conn.execute('create unique index spl_id_index ON spl(id)')
    ca_conn.execute('create unique index spl_set_id_index ON spl(set_id)')

    ca_conn.execute('create unique index nodes_id_index ON nodes(id)')
    ca_conn.execute('create index nodes_spl_id_index ON nodes(spl_id)')
    ca_conn.execute('create index nodes_parent_node_id_index ON nodes(parent_node_id)')
    ca_conn.execute('create index nodes_loc_index ON nodes(loc)')

    ca_conn.execute('create unique index sentences_id_index ON sentences(id)')
    ca_conn.execute('create index sentences_parent_node_id_index ON sentences(parent_node_id)')
    ca_conn.execute('create index sentences_spl_id_index ON sentences(spl_id)')
    ca_conn.execute('create index sentences_loc_index ON sentences(loc)')


    ca_conn.execute('create unique index codes_id_index ON codes(id)')
    ca_conn.execute('create unique index codes_code_index ON codes(code)')
    ca_conn.execute('create index codes_source_index ON codes(source)')

    ca_conn.execute('create unique index answers_id_index ON answers(id)')
    ca_conn.execute('create index answers_sentence_id_index ON answers(sentence_id)')
    ca_conn.execute('create index answers_code_id_index ON answers(code_id)')
    ca_conn.execute('create index answers_code_index ON answers(code)')
    ca_conn.execute('create index answers_true_match_index ON answers(true_match)')
    ca_conn.execute('create index answers_indication_index ON answers(indication)')
    ca_conn.execute('create index answers_negative_index ON answers(negative)')
    ca_conn.execute('create index answers_never_match_index ON answers(never_match)')
    ca_conn.execute('create index answers_dont_match ON answers(dont_match)')
    ca_conn.execute('create index answers_acronym_index ON answers(acronym)')

    ca_conn.execute('create unique index answers_hier_id_index ON answers_hier(id)')
    ca_conn.execute('create index answers_hier_child_id_index ON answers_hier(child_id)')
    ca_conn.execute('create index answers_hier_parent_id_index ON answers_hier(parent_id)')
    ca_conn.execute('create unique index answers_hier_unique_index ON answers_hier(parent_id,child_id)')


    ca_conn.execute('create unique index never_match_id_index ON never_match(id)')
    ca_conn.execute('create index never_match_spl_id_index ON never_match(spl_id)')
    ca_conn.execute('create unique index never_match_code_id_index ON never_match(code_id)')
    ca_conn.execute('create unique index never_match_unique_index ON never_match(spl_id,code_id)')

    ca_conn.execute('create unique index dont_match_id_index ON dont_match(id)')
    ca_conn.execute('create index dont_match_spl_id_index ON dont_match(spl_id)')
    ca_conn.execute('create index dont_match_code_id_index ON dont_match(code_id)')
    ca_conn.execute('create index dont_match_words_index ON dont_match(words)')
    ca_conn.execute('create unique index dont_match_unique_index ON dont_match(spl_id,code_id,words)')

    ca_conn.execute('create unique index acronym_id_index ON acronym(id)')
    ca_conn.execute('create index acronym_spl_id_index ON acronym(spl_id)')
    ca_conn.execute('create unique index acronym_words_index ON acronym(words)')
    ca_conn.execute('create unique index acronym_unique_index ON acronym(spl_id,words)')

    ca_conn.commit()




def gen_guide_spreadsheet(answer_options=['Drug indication', 'Patient background', 'Side effect', 'Not disease', 'Other']):
    global answer_guide_fn
    global working_dir
    
    writer = pd.ExcelWriter(f"{working_dir}/{answer_guide_fn}", engine='xlsxwriter')

    workbook = writer.book
    format1 = workbook.add_format({'align': 'left', 'valign': 'top', 'text_wrap': True, 'font_name': 'Calibri', 'font_size': 16})
    format2 = workbook.add_format({'align': 'left', 'valign': 'top', 'text_wrap': False, 'font_name': 'Calibri', 'font_size': 16})
    header_format = workbook.add_format({'align': 'centre', 'valign': 'top', 'text_wrap': False, 'font_name': 'Calibri', 'font_size': 16, 'bold': True})

    col_widths = [80,80]

    guide_df = pd.DataFrame([[i+1,o] for i,o in enumerate(answer_options)], columns=['id','option'])
    guide_df.to_excel(writer, index=False, sheet_name='indication')  # send df to writer
    worksheet = writer.sheets["indication"]  # pull worksheet object
    for idx, col in enumerate(guide_df):  # loop through all columns
        if idx==3: worksheet.set_column(idx, idx, col_widths[idx], format1)  # set column width
        else: worksheet.set_column(idx, idx, col_widths[idx], format2)  # set column width
    for columnnum, columnname in enumerate(list(guide_df.columns)):
        worksheet.write(0, columnnum, columnname, header_format)

    writer.save()

def set_answer_guide_fn(fn):
    global answer_guide_fn
    answer_guide_fn = fn

def gen_filter_rules_indexes():
    global ca_conn
    global donts_dict
    global nevers_dict
    global acronyms_set

    donts_dict = {}
    for spl_id,code_id,words in ca_conn.execute('select spl_id,code_id,words from dont_match'):
        code = get_code(code_id, ca_conn)  # get code
        donts_dict[code] = {'code': code, 'text': eval(words)}

    nevers_dict = {}
    for spl_id,code_id in ca_conn.execute('select spl_id,code_id from never_match'):
        code = get_code(code_id, ca_conn)  # get code
        nevers_dict[code] = {'code': code}

    acronyms_set = set()
    for spl_id,words in ca_conn.execute('select spl_id,words from acronym'):
        acronyms_set.add(tuple(sorted(eval(words))))


def set_filter_rules_fn(fn):
    global filter_rules_fn
    filter_rules_fn = fn

def set_author(new_author):
    global author
    author = new_author

def set_working_dir(d):
    global working_dir
    working_dir = d

def read_config(fn="config.ini"):
    global working_dir
    if isinstance(working_dir, str):
        config = configparser.ConfigParser()
        config.read(f"{working_dir}/{fn}")
        set_author(config['DEFAULT']['author'])
        set_working_dir(config['DEFAULT']['working_dir'])
        set_filter_rules_fn(config['DEFAULT']['filter_rules_fn'])
        set_answer_guide_fn(config['DEFAULT']['answer_guide_fn'])
    else:
        print("Set working dir")

def print_env():
    try: print(f"Author: {author}")
    except: print(f"Author: Not set")
    try: print(f"Working dir: {working_dir}")
    except: print(f"Working dir: Not set")
    try: print(f"Current SPL: {set_id} ({spl_id})")
    except: print(f"Current SPL: Not set")
    try: print(f"Filter rules filename: {filter_rules_fn}")
    except: print(f"Filter rules filename: Not set")
    try: print(f"Answer guide filename: {answer_guide_fn}")
    except: print(f"Answer guide filename: Not set")
    try: print(f"Curation answers filename: {curation_answers_fn}")
    except: print(f"Curation answers filename: Not set")

def help():
    functions = {
        "set_working_dir": ("Set global working directory","d: directory path (string)"),
        "read_config": ("Read global variables from config file","fn: file name (string)"),
        
        "set_author": ("Set the global author name","new_author: (string)"),
        "set_answer_guide_fn": ("Set global answer guide spreadsheet filename","fn: filename (string)"),
        "set_filter_rules_fn": ("Set global filter rules spreadsheet filename","fn: filename (string)"),
        "connect_dbs": ("Set up database connection global variables",""),
        "close_dbs": ("Close databases",""),
        "print_env": ("Print global variables of the curation environment",""),
        "load_indexes": ("Load indexes for grouping matches to be curated",""),
        "gen_filter_rules_indexes": ("Generate filter rule indexes, used for filtering",""),
        
        "list_spls": ("Print list of all SPLs in indications database","sort_col='id': name of sort column (int or string), ascending=False: (boolean), filter_curated=None: (boolean or None), print_results=True: (boolean), top_N=20: number of rows limit (int), c_lim=80: column width limit (ind)"),
        "select_spl": ("Select SPL from 'list_spls' output","indi_spl_id: SPL ID from list_spls() (int), verbose=False: (boolean), write_file=True: (boolean)"),
        "pick_random_spl": ("Select a random SPL from indications database","filter_curated=False: (boolean)"),
        
        "gen_guide_spreadsheet": ("Generate the answers guide spreadsheet which stores multiple choice options for curation spreadsheet verification","answer_options=['Drug indication', 'Patient background', 'Side effect', 'Not disease', 'Other']: (list)"),
        "gen_filter_rules_spreadsheet": ("Generate spreadsheet populated with filter rules",""),
        "add_filter_rules_from_spreadsheet": ("Add filter rules from the spreadsheet (allowing manual modification of the rules)","verbose=False: boolean"),
        "add_filter_rules_from_answers": ("Add filter rules from curation answers of the current SPL","verbose=False: boolean"),
        "add_filter_rules_from_all_answers": ("Add filter rules from all curation answers","verbose=False: boolean"),
	"gen_curation_spreadsheet": ("Generate curation spreadsheet",""),
        "load_curation_spreadsheet": ("Generate spreadsheet of curation answers loaded from the database",""),

        "verify_answers": ("Verify curation answers in spreadsheet, printing out errors and editting the spreadsheet to highlight errors","verbose=False: boolean, write_file=True: (boolean)"),
        "save": ("Save sentences, nodes and curation answers from spreadsheet in database","verbose=False: boolean, overwrite=True: (boolean)"),
        
        
        "delete_answers": ("Delete all curation answers for SPL in curation answers database",""),
        "delete_sentences": ("Delete all curation answers and sentence data for SPL in curation answers database",""),
        "delete_spl": ("Delete all data for SPL in curation answers database",""),
        "delete_filter_rules": ("Delete filter rules derived from the current SPL",""),
        "delete_all_filter_rules": ("Delete all filter rules from database",""),
        
        "search_term": ("Search for disease indication ontology entities by name","q: query (string), c_lim=60: column width limit (int), top_N=20: number of rows limit (int), print_results=True: (boolean)"),
        "search_code": ("Search for disease indication ontology entities by partial ID","q: query (string), c_lim=60: column width limit (int), top_N=20: number of rows limit (int), print_results=True: (boolean)"),
        "closest_efo": ("List closest EFO entities","q: query (string)"), 
    }
    
    data = [[k,vs[0],vs[1]] for k,vs in functions.items()]
    
    # string wrapping
    col_widths = [40,60,100]
    for i,r in enumerate(data):
        for j,d in enumerate(r):
            new_d = ""
            while True:
                new_d += d[:col_widths[j]]
                d = d[col_widths[j]:]
                if len(d)>0: new_d += "\n"
                else: break
            data[i][j] = new_d
    
    p_df = pd.DataFrame(data, columns=["Function", "Description", "Parameters"])
    print(tabulate(p_df, showindex=False, tablefmt='plain', colalign=("right", "left")))  # headers='keys'



def expand_sentence(original_sentence):
    expanded_sentence = tq.expand_lists(copy.deepcopy(original_sentence))
    expanded_sentence = tq.expand_brackets(expanded_sentence)
    expanded_sentence = tq.expand_slash(expanded_sentence)
    expanded_sentence = tq.expand_hyphen(expanded_sentence)
    expanded_sentence = tq.expand_index(original_sentence, tq.query_thesauruses, tq.query_names_indexes, [thes_conn, {'ncit', 'uberon'}], [thes_conn])  # substitute based on thesaurus
    return expanded_sentence

def gen_matches(expanded_sentence):
    matches = set()
    for m in tq.query_sentence(expanded_sentence, index_conn):
        original_locs = set()
        for path in m['paths']:
            original_path = set()
            for l in path:
                original_path.update(gm.rec_fetch_parent(l, expanded_sentence))
            original_locs.add(tuple(original_path))
        matches.add((m['onto_id'], m['predicate_type'], tuple(original_locs)))
    
    return matches

def gen_spl_matches():
    global indi_conn
    global set_id    
    
    indi_spl_id = int(list(indi_conn.execute("select id from spl where set_id=?",(set_id,)))[0][0])
    
    sentences = []
    matches_index = {}
    for s_id, parent_node_id, loc, string, sentence, expanded_sentence in indi_conn.execute("select id, parent_node_id, loc, string, sentence, expanded_sentence from sentences where spl_id=?", (indi_spl_id,)):
        sentence = eval(sentence)
        loc = eval(loc)
        expanded_sentence = expand_sentence(sentence)
        matches = gen_matches(expanded_sentence)
        matches_index[s_id] = matches
        sentences.append([s_id, loc, sentence, expanded_sentence])
    
    return sentences, matches_index

def gen_sentences_df(sentences, matches_index):
    sentences_data = []
    sentence_index = {}
    for s_id, loc, sentence, expanded_sentence in sentences:
        matches = matches_index[s_id]
        s = []
        n_string = []
        for w in sentence['words'].values():
            if 'parent' in w.keys():
                if not w['id'] == w['parent']: continue
            n_string.append(f"{w['word']}|{w['id']}")
            s.append(w['word'])
        sentence_index[loc] = {'id': s_id, 'text': ' '.join(n_string)}

        sentences_data.append([loc, s_id, ' '.join(n_string), ' '.join(["  " for i in range(len(loc))] + s)])

    sentences_df = pd.DataFrame(sentences_data, columns=["loc", "sentence_id", "enumerated_string", "string"])
    
    return sentences_df, sentence_index

def filter_match_sentences(sentences, matches_index):
    global donts_dict
    global nevers_dict
    global acronyms_set
    
    filtered_sentences = []
    filtered_matches_index = {}
    for s_id, loc, sentence, expanded_sentence in sentences:
        matches = matches_index[s_id]
        filtered_matches = []
        for m in matches:
            code = m[0]
            p_type = m[1]
            match_paths = m[2]

            # check never rules
            if code in nevers_dict.keys():
                continue

            # check acronym rule
            acronym_filtered_match_paths = set()
            for mp in match_paths:
                mp_words = [sentence['words'][i]['word'] for i in mp]
                if all([w.isupper() for w in mp_words]):
                    acronym_filtered_match_paths.add(mp)
                else:
                    if tuple(sorted([w.lower() for w in mp_words])) in acronyms_set:
                        continue
                    else:
                        acronym_filtered_match_paths.add(mp)

            # check don't rules
            dont_filtered_match_paths = set()
            if code in donts_dict.keys():
                text = {w.lower() for w in donts_dict[code]['text']}
                for mp in match_paths:
                    mp_words = [sentence['words'][i]['word'] for i in mp]
                    mp_word_set = {w.lower() for w in mp_words}
                    if (len(mp_word_set)==len(text)) and (len(mp_word_set-text)==0):
                        continue
                    else:
                        dont_filtered_match_paths.add(mp)
            else:
                dont_filtered_match_paths.update(match_paths)

            filtered_match_paths = dont_filtered_match_paths & acronym_filtered_match_paths
            if len(filtered_match_paths)>0:
                filtered_matches.append((code,p_type,tuple(sorted(filtered_match_paths))))

        if len(filtered_matches)>0:
            filtered_sentences.append((s_id, loc, sentence, expanded_sentence))
            filtered_matches_index[s_id] = filtered_matches
    return filtered_sentences, filtered_matches_index

def group_matches(sentences, matches):
    joined_sentences = [tuple(list(s)+[matches[s[0]]]) for s in sentences]
    condensed_matches, condensed_matches_r = gm.get_doc_condensed_matches(  joined_sentences, 
                                                                            indi_conn, 
                                                                            index_conn, 
                                                                            equivalent_entities_groups_index, 
                                                                            equivalent_entities_groups_index_r, 
                                                                            disease_hierarchy_index, 
                                                                            disease_hierarchy_distance_index, 
                                                                            rev_disease_hierarchy_distance_index)

    response = {'condensed_matches': condensed_matches, 'condensed_matches_r': condensed_matches_r}
    
    return condensed_matches

def gen_blank_answers_data(condensed_matches, sentence_index):
    data = []
    group_i = 0
    for i,d in enumerate(condensed_matches):
        for j,matches in d['match_groups'].items():
            group_i+=1

            for code, p_type, name in matches:
                s_id = sentence_index[d['loc']['sentence_loc']]['id']
                s_text = sentence_index[d['loc']['sentence_loc']]['text']

                path = d['loc']['path']
                data.append([group_i, s_id, s_text, path, code, name, p_type, "?", "", "", "", "", "", ""])  # set_id, d['loc']['sentence_loc']
    
    data = sorted(data, key=lambda x:(int(x[1]),int(x[3][0]),x[0]))

    current_s_id = -1
    for i,d in enumerate(data):
        if current_s_id == d[1]: d[2] = ""
        else: current_s_id = d[1]
        d[3] = ','.join([str(i) for i in d[3]])
        data[i] = [i]+d
    
    return data

def write_spreadsheet_answers(answers_df, writer):
    col_widths = [8,16,18,100,16,26,22,34,18,18,18,18,18,16,60]

    answers_df.to_excel(writer, index=False, sheet_name='answers')  # send df to writer

    workbook = writer.book
    format1 = workbook.add_format({'align': 'left', 'valign': 'top', 'text_wrap': True, 'font_name': 'Calibri', 'font_size': 16})
    format2 = workbook.add_format({'align': 'left', 'valign': 'top', 'text_wrap': False, 'font_name': 'Calibri', 'font_size': 16})
    header_format = workbook.add_format({'align': 'centre', 'valign': 'top', 'text_wrap': False, 'font_name': 'Calibri', 'font_size': 16, 'bold': True})

    worksheet = writer.sheets["answers"]  # pull worksheet object
    for idx, col in enumerate(answers_df):  # loop through all columns
        worksheet.set_column(idx, idx, col_widths[idx], format1)  # set column width

    for columnnum, columnname in enumerate(list(answers_df.columns)):
        worksheet.write(0, columnnum, columnname, header_format)
    
def write_spreadsheet_sentences(sentences_df, writer):
    col_widths = [30,16,25,300]
    
    workbook = writer.book
    format1 = workbook.add_format({'align': 'left', 'valign': 'top', 'text_wrap': True, 'font_name': 'Calibri', 'font_size': 16})
    format2 = workbook.add_format({'align': 'left', 'valign': 'top', 'text_wrap': False, 'font_name': 'Calibri', 'font_size': 16})
    header_format = workbook.add_format({'align': 'centre', 'valign': 'top', 'text_wrap': False, 'font_name': 'Calibri', 'font_size': 16, 'bold': True})
    
    sentences_df.to_excel(writer, index=False, sheet_name='sentences')  # send df to writer
    worksheet = writer.sheets["sentences"]  # pull worksheet object
    for idx, col in enumerate(sentences_df):  # loop through all columns
        if idx==3: worksheet.set_column(idx, idx, col_widths[idx], format1)  # set column width
        else: worksheet.set_column(idx, idx, col_widths[idx], format2)  # set column width
    for columnnum, columnname in enumerate(list(sentences_df.columns)):
        worksheet.write(0, columnnum, columnname, header_format)
        
def write_spreadsheet_guide(guide_df, writer):
    col_widths = [6,30]
    
    workbook = writer.book
    format1 = workbook.add_format({'align': 'left', 'valign': 'top', 'text_wrap': True, 'font_name': 'Calibri', 'font_size': 16})
    format2 = workbook.add_format({'align': 'left', 'valign': 'top', 'text_wrap': False, 'font_name': 'Calibri', 'font_size': 16})
    header_format = workbook.add_format({'align': 'centre', 'valign': 'top', 'text_wrap': False, 'font_name': 'Calibri', 'font_size': 16, 'bold': True})
    
    guide_df.to_excel(writer, index=False, sheet_name='answer_guide')  # send df to writer
    worksheet = writer.sheets["answer_guide"]  # pull worksheet object
    for idx, col in enumerate(guide_df):  # loop through all columns
        if idx==3: worksheet.set_column(idx, idx, col_widths[idx], format1)  # set column width
        else: worksheet.set_column(idx, idx, col_widths[idx], format2)  # set column width
    for columnnum, columnname in enumerate(list(guide_df.columns)):
        worksheet.write(0, columnnum, columnname, header_format)

def gen_curation_spreadsheet():
    global set_id
    global curation_answers_fn
    global answer_guide_fn
    global working_dir
    global sentences
    
    sentences, matches = gen_spl_matches()
    sentences_df, sentence_index = gen_sentences_df(sentences, matches)
    
    filtered_sentences, filtered_matches = filter_match_sentences(sentences, matches)
    condensed_matches = group_matches(filtered_sentences, matches)
    data = gen_blank_answers_data(condensed_matches, sentence_index)
    answers_df = pd.DataFrame(data, columns=['id', 'group_id', 'sentence_id', 'sentence', 'match_path', 'code', 'name', 'predicate_type', "true_match", "indication", "acronym", "never_match", "dont_match", "negative", "note"])  # 'set_id', 'sentence_loc'
    
    guide_df = pd.read_excel(f"{working_dir}/{answer_guide_fn}", sheet_name='indication')
    
    writer = pd.ExcelWriter(f"{working_dir}/{curation_answers_fn}", engine='xlsxwriter')

    write_spreadsheet_answers(answers_df, writer)
    write_spreadsheet_sentences(sentences_df, writer)
    write_spreadsheet_guide(guide_df, writer)

    writer.save()
    writer.close()

    save_sentences(verbose=False, overwrite=True)

def gen_populated_answers_data(sentence_index):
    global ca_conn

    s_ids = tuple([v['id'] for k,v in sentence_index.items()])

    data = []
    for answer_id, group_id, sentence_id, locs, code, predicate_type, code_id, true_match, negative, indication, never_match, dont_match, acronym, note, timestamp, author in ca_conn.execute(f'select answer_id, group_id, sentence_id, locs, code, predicate_type, code_id, true_match, negative, indication, never_match, dont_match, acronym, note, timestamp, author from answers where sentence_id in {repr(s_ids)}'):
        if true_match is None: 
            true_match = "?"
        else:
            true_match = "X" if true_match else ""
        negative = "X" if negative else ""
        never_match = "X" if never_match else ""
        dont_match = "X" if dont_match else ""
        acronym = "X" if acronym else ""
        locs = eval(locs)
        s_loc = eval(list(ca_conn.execute('select loc from sentences where id=?', (sentence_id,)))[0][0])
        s_text = sentence_index[s_loc]['text']
        try: name = list(ca_conn.execute('select name from codes where id=?', (code_id,)))[0][0]  # fetch code name
        except: name = ""
        data.append([answer_id, group_id, sentence_id, s_text, locs, code, name, predicate_type, true_match, negative, indication, never_match, dont_match, acronym, note])  # set_id, d['loc']['sentence_loc']
    
    data = sorted(data, key=lambda x:(int(x[2]),int(x[4][0]),x[1]))

    current_s_id = -1
    for i,d in enumerate(data):
        if current_s_id == d[2]: d[3] = ""
        else: current_s_id = d[2]
        d[4] = ','.join([str(i) for i in d[4]])
    
    return data

def load_sentences():
    global ca_conn
    global spl_id

    sentences = []
    sentence_index = {}
    for s_id, loc, s_text, sentence, expanded_sentence in ca_conn.execute('select id, loc, string, sentence, expanded_sentence from sentences where spl_id=?', (spl_id,)):
        loc = eval(loc)
        sentence = eval(sentence)
        expanded_sentence = eval(expanded_sentence)
        sentences.append([s_id, loc, sentence, expanded_sentence])

        n_string = []
        for w in sentence['words'].values():
            if 'parent' in w.keys():
                if not w['id'] == w['parent']: continue
            n_string.append(f"{w['word']}|{w['id']}")

        sentence_index[loc] = {'id': s_id, 'text': ' '.join(n_string)}

    return sentences, sentence_index

def gen_populated_sentences_df():
    global spl_id
    global ca_conn

    sentences, sentence_index = load_sentences()

    sentences_data = []
    for s_id, loc, sentence, expanded_sentence in sentences:
        s = []
        n_string = []
        for w in sentence['words'].values():
            if 'parent' in w.keys():
                if not w['id'] == w['parent']: continue
            n_string.append(f"{w['word']}|{w['id']}")
            s.append(w['word'])
        sentence_index[loc] = {'id': s_id, 'text': ' '.join(n_string)}

        sentences_data.append([loc, s_id, ' '.join(n_string), ' '.join(["  " for i in range(len(loc))] + s)])

    sentences_df = pd.DataFrame(sentences_data, columns=["loc", "sentence_id", "enumerated_string", "string"])
    
    return sentences, sentences_df, sentence_index
    
def load_curation_spreadsheet():
    global sentences

    sentences, sentences_df, sentence_index = gen_populated_sentences_df()
    data = gen_populated_answers_data(sentence_index)
    answers_df = pd.DataFrame(data, columns=['id', 'group_id', 'sentence_id', 'sentence', 'match_path', 'code', 'name', 'predicate_type', "true_match", "indication", "acronym", "never_match", "dont_match", "negative", "note"])  # 'set_id', 'sentence_loc'
    guide_df = pd.read_excel(f"{working_dir}/{answer_guide_fn}", sheet_name='indication')
    
    writer = pd.ExcelWriter(f"{working_dir}/{curation_answers_fn}", engine='xlsxwriter')
    write_spreadsheet_answers(answers_df, writer)
    write_spreadsheet_sentences(sentences_df, writer)
    write_spreadsheet_guide(guide_df, writer)
    writer.save()
    writer.close()




def parse_answer_row(r, answers_df):
    r = {c:r[i] for i,c in enumerate(answers_df.columns)}
    
    try: r['match_path'] = tuple([int(n) for n in r['match_path'].split(',')])
    except Exception as e: 
        if isinstance(r['match_path'], int):
            r['match_path'] = [r['match_path']]
        else:
            raise e
    
    if pd.isna(r['true_match']):
        r['true_match'] = False
    else:
        if r['true_match'] == '?':
            r['true_match'] = None
        else:
            r['true_match'] = True
    
    if pd.isna(r['sentence']): r['sentence'] = ""
    if pd.isna(r['name']): r['name'] = ""
    if pd.isna(r['indication']): r['indication'] = None
    r['acronym'] = False if (pd.isna(r['acronym']) or len(r['acronym'])==0) else True
    r['never_match'] = False if (pd.isna(r['never_match']) or len(r['never_match'])==0) else True
    r['dont_match'] = False if (pd.isna(r['dont_match']) or len(r['dont_match'])==0)else True
    r['negative'] = False if (pd.isna(r['negative']) or len(r['negative'])==0) else True
    if pd.isna(r['note']): r['note'] = ""
    
    return r

def read_answers_spreadsheet():
    global curation_answers_fn
    global answer_guide_fn
    global working_dir
    global indi_conn
    
    answers_df = pd.read_excel(f"{working_dir}/{curation_answers_fn}", sheet_name='answers')

    guide_df = pd.read_excel(f"{working_dir}/{answer_guide_fn}", sheet_name='indication')
    indication_guide = {r[0]:r[1] for r in guide_df.values}
    indication_guide_r = {tuple([v.lower() for v in vs.split(' ')]):k for k,vs in indication_guide.items()}

    r_data = []
    for row in answers_df.values:
        r = parse_answer_row(row, answers_df)
        r['error'] = set()

        if r['true_match'] is None:
            r['error'].add("unanswered")

        # look up sentence in database
        try:
            sentence = eval(list(indi_conn.execute("select sentence from sentences where id=?", (r['sentence_id'],)))[0][0])
            if not all([i in sentence['words'].keys() for i in r['match_path']]):
                r['error'].add("word not in sentence")
        except:
            r['error'].add("sentence not found")

        # look up match code in index
        # if not r['code'] in code_name_index.keys():
        #     r['error'].add("match code not found")

        # check that 'indication' field is valid, and normalise the value
        if isinstance(r['indication'], int):
            if not r['indication'] in indication_guide.keys():
                r['error'].add("indication answer number not valid")
            else:
                r['indication'] = indication_guide[r['indication']]
        else:
            if r['indication'] is None:
                if r['true_match'] is True:
                    r['error'].add("indication not present")
            else:
                ind = tuple([v.lower() for v in r['indication'].split(' ')])
                if ind in indication_guide_r.keys():
                    r['indication'] = indication_guide[indication_guide_r[ind]]
                else:
                    r['error'].add("indication answer text not valid")

        r_data.append(r)
    
    return r_data

# generate normalised spreadsheet
def gen_answers_data(r_data):
    data = []
    errors = {}
    for i,r in enumerate(r_data):
        if r['true_match']:
            true_match = "X"
        else:
            if r['true_match'] is None:
                true_match = "?"
            else:
                true_match = ""
        
        path = ','.join([str(i) for i in r['match_path']])
        
        data.append([r['id'], 
                     r['group_id'], 
                     r['sentence_id'], 
                     r['sentence'], 
                     path, 
                     r['code'], 
                     r['name'], 
                     r['predicate_type'], 
                     true_match, 
                     r['indication'], 
                     "X" if r['acronym'] else "", 
                     "X" if r['never_match'] else "", 
                     "X" if r['dont_match'] else "", 
                     "X" if r['negative'] else "", 
                     r['note']])  # set_id, d['loc']['sentence_loc']
        
        if len(r['error'])>0:
            errors[i] = set()
            if "indication answer text not valid" in r['error']:
                errors[i].add(9)
            if "indication not present" in r['error']:
                errors[i].add(9)
            if "indication answer number not valid" in r['error']:
                errors[i].add(9)
            # if "match code not found" in r['error']:
            #     errors[i].add(5)
            if "sentence not found" in r['error']:
                errors[i].add(2)
            if "word not in sentence" in r['error']:
                errors[i].add(4)
            if "unanswered" in r['error']:
                errors[i].add(8)
        
    return data, errors

def write_spreadsheet_answers_errors(answers_df, errors):
    global curation_answers_fn
    global working_dir
    
    sentences_df = pd.read_excel(f"{working_dir}/{curation_answers_fn}", sheet_name='sentences')
    guide_df = pd.read_excel(f"{working_dir}/{curation_answers_fn}", sheet_name='answer_guide')

    writer = pd.ExcelWriter(f"{working_dir}/{curation_answers_fn}", engine='xlsxwriter')
    answers_df.to_excel(writer, index=False, sheet_name='answers')  # send df to writer
    
    workbook = writer.book
    format_dict = {'align': 'left', 'valign': 'top', 'font_name': 'Calibri', 'font_size': 16}
    format1 = workbook.add_format({**format_dict, **{'text_wrap': False}})
    format2 = workbook.add_format({**format_dict, **{'text_wrap': True}})
    header_format = workbook.add_format({**format_dict, **{'bold': True}})
    error_row_format1 = workbook.add_format({**format_dict, **{'text_wrap': False, 'bg_color': '#fcd4cf'}})
    error_row_format2 = workbook.add_format({**format_dict, **{'text_wrap': True, 'bg_color': '#fcd4cf'}})
    error_cell_format1 = workbook.add_format({**format_dict, **{'text_wrap': False, 'bg_color': '#ffa296', 'bottom':5, 'top':5, 'left':5, 'right':5}})
    error_cell_format2 = workbook.add_format({**format_dict, **{'text_wrap': True, 'bg_color': '#ffa296', 'bottom':5, 'top':5, 'left':5, 'right':5}})
    
    worksheet = writer.sheets["answers"]  # pull worksheet object
    col_index = {c:i for i,c in enumerate(answers_df.columns)}
    for idx, col in enumerate(answers_df):  # loop through all columns
        for idy, v in enumerate(answers_df[col]):
            if idy in errors.keys():
                if idx in errors[idy]:
                    worksheet.write(idy+1, idx, v, error_cell_format2)
                else:
                    worksheet.write(idy+1, idx, v, error_row_format2)
            else:
                worksheet.write(idy+1, idx, v, format2)
    
    col_widths = [8,16,18,100,16,26,22,34,18,18,18,18,18,16,60]
    for idx, col in enumerate(answers_df):  # loop through all columns
        worksheet.set_column(idx, idx, col_widths[idx])  # set column width
        
    for columnnum, columnname in enumerate(list(answers_df.columns)):
        worksheet.write(0, columnnum, columnname, header_format)
    
    write_spreadsheet_sentences(sentences_df, writer)
    write_spreadsheet_guide(guide_df, writer)

    writer.save()
    writer.close()

def verify_answers(verbose=False, write_file=True):
    r_data = read_answers_spreadsheet()
    data, errors = gen_answers_data(r_data)
    answers_df = pd.DataFrame(data, columns=['id', 'group_id', 'sentence_id', 'sentence', 'match_path', 'code', 'name', 'predicate_type', "true_match", "indication", "acronym", "never_match", "dont_match", "negative", "note"])  # 'set_id', 'sentence_loc'
    
    if write_file:
        write_spreadsheet_answers_errors(answers_df, errors)
    
    if verbose:
        # summarise results
        error_row_count = len([r for r in r_data if len(r['error'])>0])
        print(f"--- {error_row_count} / {len(r_data)} rows have errors ---")
        for r in r_data:
            if len(r['error'])>0:
                print(r['id'], '; '.join(r['error']))



def add_spl(verbose=False, overwrite=True):
    global zip_metadata
    global set_id
    global ca_conn
    global indi_conn
    
    d = zip_metadata[set_id]
    try:
        c = ca_conn.execute("insert into spl(set_id,date,version,title) values (?,?,?,?)", (set_id,d['date'],d['spl_version'],d['title']))
        spl_id = c.lastrowid
    except IntegrityError as e:
        spl_id = int(list(ca_conn.execute("select id from spl where set_id=?", (set_id,)))[0][0])
        if overwrite:
            c = ca_conn.execute("replace into spl(id,set_id,date,version,title) values (?,?,?,?,?)", (spl_id,set_id,d['date'],d['spl_version'],d['title']))
            if verbose: print(f"SPL {set_id} overwritten")
        else:
            if verbose: print(f"SPL {set_id} already in database")
    
    return spl_id

def save_nodes(verbose=False, overwrite=True):
    global set_id
    global ca_conn
    global indi_conn
    
    indi_spl_id = int(list(indi_conn.execute('select id from spl where set_id=?', (set_id,)))[0][0])  # get spl_id
    for node_id, parent_node_id, tag, loc in indi_conn.execute("select id, parent_node_id, tag, loc from nodes where spl_id=?", (indi_spl_id,)):
        try:
            ca_conn.execute('insert into nodes(id, spl_id, parent_node_id, tag, loc) values (?,?,?,?,?)', (node_id, spl_id, parent_node_id, tag, loc))
        except IntegrityError as e:
            if overwrite:
                ca_conn.execute('replace into nodes(id, spl_id, parent_node_id, tag, loc) values (?,?,?,?,?)', (node_id, spl_id, parent_node_id, tag, loc))
                if verbose: print(f"Node {node_id} overwritten")
            else:
                if verbose: print(f"Node {node_id} already in database")
    ca_conn.commit()

def save_sentences(verbose=False, overwrite=True):
    global ca_conn
    global sentences

    for s_id, loc, sentence, expanded_sentence in sentences:
        parent_node_id, string = list(indi_conn.execute("select parent_node_id, string from sentences where id=?", (s_id,)))[0]  # get parent_id and string from indi_conn database
        try: 
            ca_conn.execute("insert into sentences(id,parent_node_id,spl_id,loc,string,sentence,expanded_sentence) values (?,?,?,?,?,?,?)", (s_id, int(parent_node_id), spl_id, repr(loc), string, repr(sentence), repr(expanded_sentence)))
        except IntegrityError as e:
            if overwrite:
                ca_conn.execute("replace into sentences(id,parent_node_id,spl_id,loc,string,sentence,expanded_sentence) values (?,?,?,?,?,?,?)", (s_id, int(parent_node_id), spl_id, repr(loc), string, repr(sentence), repr(expanded_sentence)))
                if verbose: print(f"Sentence {s_id} overwritten")
            else:
                if verbose: print(f"Sentence {s_id} already in database")
    ca_conn.commit()

def save(verbose=False, overwrite=True):
    save_nodes(verbose=verbose, overwrite=overwrite)
    save_sentences(verbose=verbose, overwrite=overwrite)
    save_curation_answers(verbose=verbose, overwrite=overwrite)

def insert_code(code, verbose=False, overwrite=True):
    global ca_conn
    global index_conn
    
    # look up code in index_conn
    try: 
        code_id, predicate_type, source, string = list(index_conn.execute('select id, predicate_type, source, string from strings where onto_id=?', (code,)))[0]
    except:
        if verbose: print(f"Code '{code}' not found in database")
        source = ""
        string = ""
    
    try: 
        predicate_type, name = code_name_index[code]
    except:
        name = string
        if verbose: print(f"Name for '{code}' not found in index")
    
    # try to insert into database, overwrite if requested
    try: 
        c = ca_conn.execute('insert into codes(code, source, name) values (?,?,?)', (code,source,name))
        code_id = int(c.lastrowid)
    except IntegrityError as e:
        if overwrite:
            c = ca_conn.execute('replace into codes(code, source, name) values (?,?,?)', (code,source,name))
            code_id = int(c.lastrowid)
            if verbose: print(f"Code '{code}' overwritten")
        else:
            code_id = list(ca_conn.execute('select id from codes where code=?', (code,)))[0][0]
            if verbose: print(f"Code '{code}' already in database")
                
    return code_id

# answer data
def save_curation_answers(verbose=False, overwrite=True):
    global ca_conn
    global author
    global curation_answers_fn
    global working_dir
    
    answers_df = pd.read_excel(f"{working_dir}/{curation_answers_fn}", sheet_name='answers')
    
    hier_dict = {}
    group_dict = {}
    for r in answers_df.values:
        r = parse_answer_row(r, answers_df)
        ts = str(datetime.now())
        code_id = insert_code(r['code'], verbose=False, overwrite=False)
        row_data = (r['id'],
                    r['group_id'],
                    r['sentence_id'],
                    repr(r['match_path']),
                    r['code'],
                    r['predicate_type'],
                    code_id,
                    r['true_match'],
                    r['negative'],
                    r['indication'],
                    r['never_match'],
                    r['dont_match'],
                    r['acronym'],
                    r['note'],
                    ts,
                    author)
        
        try: parent_id = group_dict[r['group_id']]
        except: 
            parent_id = r['id']
            group_dict[r['group_id']] = parent_id    
        try: hier_dict[parent_id].add(r['id'])
        except: hier_dict[parent_id] = {r['id']}
            
        try:
            ca_conn.execute('insert into answers(answer_id, group_id, sentence_id, locs, code, predicate_type, code_id, true_match, negative, indication, never_match, dont_match, acronym, note, timestamp, author) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', row_data)
        except IntegrityError as e:
            if overwrite:
                ca_conn.execute('replace into answers(answer_id, group_id, sentence_id, locs, code, predicate_type, code_id, true_match, negative, indication, never_match, dont_match, acronym, note, timestamp, author) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', row_data)
                if verbose: print(f"Answer {row_data[0]} overwritten")
            else:
                if verbose: print(f"Answer {row_data[0]} already in database")
    
    # add answer hierarchy from group_id
    for parent_id, child_ids in hier_dict.items():
        for child_id in child_ids:
            try: ca_conn.execute('insert into answers_hier(parent_id, child_id) values (?,?)', (parent_id,child_id))
            except IntegrityError as e: continue
    
    ca_conn.commit()



def get_ids():
    global ca_conn
    global spl_id
    
    #spl_id = int(list(ca_conn.execute('select id from spl where set_id=?', (set_id,)))[0][0])  # get spl_id
    sentence_ids = [int(i[0]) for i in ca_conn.execute('select id from sentences where spl_id=?', (spl_id,))]  # get sentence IDs
    node_ids = [int(i[0]) for i in ca_conn.execute('select id from nodes where spl_id=?', (spl_id,))]  # get node IDs
    
    # get answers/answers_hier
    answer_ids = [int(i[0]) for i in ca_conn.execute(f'select id from answers where sentence_id in {repr(tuple(sentence_ids))}')]
    answers_hier_ids = [int(i[0]) for i in ca_conn.execute(f'select id from answers_hier where parent_id in {repr(tuple(answer_ids))}')]
    
    return sentence_ids, node_ids, answer_ids, answers_hier_ids  # spl_id

def delete_curation_answers_queries(answers_hier_ids, answer_ids):
    ca_conn.execute(f'delete from answers_hier where id in {repr(tuple(answers_hier_ids))}')
    ca_conn.execute(f'delete from answers where id in {repr(tuple(answer_ids))}')
    
def delete_sentences_queries(sentence_ids, node_ids):
    ca_conn.execute(f'delete from sentences where id in {repr(tuple(sentence_ids))}')
    ca_conn.execute(f'delete from nodes where id in {repr(tuple(node_ids))}')

def delete_spl_queries(spl_id):
    ca_conn.execute(f'delete from spl where id=?', (spl_id,))
    
def delete_answers():
    global ca_conn
    sentence_ids, node_ids, answer_ids, answers_hier_ids = get_ids()
    delete_curation_answers_queries(answers_hier_ids, answer_ids)
    ca_conn.commit()
    
def delete_sentences():
    global ca_conn
    sentence_ids, node_ids, answer_ids, answers_hier_ids = get_ids()
    delete_curation_answers_queries(answers_hier_ids, answer_ids)
    delete_sentences_queries(sentence_ids, node_ids)
    ca_conn.commit()
    
def delete_spl():
    global ca_conn
    sentence_ids, node_ids, answer_ids, answers_hier_ids = get_ids()
    delete_curation_answers_queries(answers_hier_ids, answer_ids)
    delete_sentences_queries(sentence_ids, node_ids)
    delete_spl_queries(spl_id)
    ca_conn.commit()

def delete_filter_rules():
    global ca_conn
    global spl_id
    
    ca_conn.execute('delete from acronym where spl_id=?', (spl_id,))
    ca_conn.execute('delete from dont_match where spl_id=?', (spl_id,))
    ca_conn.execute('delete from never_match where spl_id=?', (spl_id,))
    
    ca_conn.commit()

def delete_all_filter_rules():
    global ca_conn
    
    ca_conn.execute('delete from acronym')
    ca_conn.execute('delete from dont_match')
    ca_conn.execute('delete from never_match')
    
    ca_conn.commit()

def add_dont_match(spl_id, code, words, verbose, ca_conn):
    try: code_id = int(list(ca_conn.execute('select id from codes where code=?', (code,)))[0][0])
    except: code_id = insert_code(code, verbose=verbose, overwrite=False)
    try:
        ca_conn.execute('insert into dont_match(spl_id,code_id,words) values(?,?,?)', (spl_id, code_id, repr(words)))
    except IntegrityError as e:
        if verbose: print(f"Don't-match rule '{code}' / '{repr(words)}' already in database")

def add_acronym(spl_id, words, verbose, ca_conn):        
    try:
        ca_conn.execute('insert into acronym(spl_id,words) values(?,?)', (spl_id, repr(words)))
    except IntegrityError as e:
        if verbose: print(f"Acronym rule '{repr(words)}' already in database")
            
def add_never_match(spl_id, code, verbose, ca_conn):
    try: code_id = int(list(ca_conn.execute('select id from codes where code=?', (code,)))[0][0])
    except: code_id = insert_code(code, verbose=verbose, overwrite=False)
    try:
        ca_conn.execute('insert into never_match(spl_id,code_id) values(?,?)', (spl_id, code_id))
    except IntegrityError as e:
        if verbose: print(f"Never-match rule '{code}' already in database")
            
def add_filter_rules_from_answer(spl_id, s_id, match_path, code, acronym, never_match, dont_match, verbose=False):
    global ca_conn

    match_path = eval(match_path)
    acronym = True if acronym>0 else False
    never_match = True if never_match>0 else False
    dont_match = True if dont_match>0 else False

    # if dont_match or acronym then fetch sentence words
    if dont_match or acronym:
        sentence = eval(list(ca_conn.execute(f'select sentence from sentences where id=?', (s_id,)))[0][0])
        words = [sentence['words'][i]['word'] for i in match_path]
        
    if dont_match: add_dont_match(spl_id, code, words, verbose, ca_conn)
    if acronym: add_acronym(spl_id, words, verbose, ca_conn)
    if never_match: add_never_match(spl_id, code, verbose, ca_conn)

def add_filter_rules_from_answers(verbose=False):
    global ca_conn
    global spl_id
    
    sentence_ids, node_ids, answer_ids, answers_hier_ids = get_ids()
    for s_id, match_path, code, acronym, never_match, dont_match in ca_conn.execute(f'select sentence_id,locs,code,acronym,never_match,dont_match from answers where id in {repr(tuple(answer_ids))}'):
        add_filter_rules_from_answer(spl_id, s_id, match_path, code, acronym, never_match, dont_match, verbose=verbose)
    
    ca_conn.commit()
    
def add_filter_rules_from_all_answers(verbose=False):
    global ca_conn
    
    for s_id, match_path, code, acronym, never_match, dont_match in ca_conn.execute(f'select sentence_id,locs,code,acronym,never_match,dont_match from answers'):
        spl_id = int(list(ca_conn.execute('select spl_id from sentences where id=?', (s_id,)))[0][0])
        add_filter_rules_from_answer(spl_id, s_id, match_path, code, acronym, never_match, dont_match, verbose=verbose)
    
    ca_conn.commit()
			    
def get_insert_spl_id(set_id, ca_conn, cache, verbose=False):
    global zip_metadata

    try: return cache[set_id], cache
    except:
        try: spl_id = int(list(ca_conn.execute('select id from spl where set_id=?', (set_id,)))[0][0])  # get spl_id
        except:  # add SPL to database and get ID
            try: 
                m = zip_metadata[set_id]
                date = m['date']
                version = m['spl_version']
                title = m['title']
            except:
                date = ""
                version = ""
                title = ""
            c = ca_conn.execute('insert into spl(set_id,title,date,version) values (?,?,?,?)', (set_id,title,date,version))
            spl_id = c.lastrowid  # set global spl_id
            if verbose: print(f'SPL {set_id} added to database with ID: {spl_id}')
        cache[set_id] = spl_id
        return spl_id, cache

def get_spl_id(set_id, ca_conn, cache):
    try: return cache[set_id], cache
    except:
        try: spl_id = int(list(ca_conn.execute('select id from spl where set_id=?', (set_id,)))[0][0])  # get spl_id
        except:  # add SPL to database and get ID
            spl_id = False
        cache[set_id] = spl_id
        return spl_id, cache

def get_code(code_id, ca_conn):
    try: code = list(ca_conn.execute('select code from codes where id=?', (code_id,)))[0][0]  # get code
    except: code = False
    return code
    
def get_set_id(spl_id, ca_conn):
    try: set_id = list(ca_conn.execute('select set_id from spl where id=?', (spl_id,)))[0][0]  # get set_id
    except: set_id = False
    return set_id

def add_filter_rules_from_spreadsheet(verbose=False):
    global ca_conn
    global filter_rules_fn
    global working_dir
    
    # load list data
    acronyms_df = pd.read_excel(f"{working_dir}/{filter_rules_fn}", sheet_name='acronyms')
    nevers_df = pd.read_excel(f"{working_dir}/{filter_rules_fn}", sheet_name='nevers')
    donts_df = pd.read_excel(f"{working_dir}/{filter_rules_fn}", sheet_name='donts')
    
    cache = {}
    for set_id, words in acronyms_df.values:
        words = eval(words)
        spl_id, cache = get_insert_spl_id(set_id, ca_conn, cache, verbose=verbose)
        add_acronym(spl_id, words, verbose, ca_conn)
    for set_id, code in nevers_df.values:
        spl_id, cache = get_insert_spl_id(set_id, ca_conn, cache, verbose=verbose)
        add_never_match(spl_id, code, verbose, ca_conn)
    for set_id, code, words in donts_df.values:
        words = eval(words)
        spl_id, cache = get_insert_spl_id(set_id, ca_conn, cache, verbose=verbose)
        add_dont_match(spl_id, code, words, verbose, ca_conn)
        
    ca_conn.commit()

def gen_filter_rules_spreadsheet():
    global ca_conn
    global filter_rules_fn
    global working_dir
    
    acronyms_data = []
    for spl_id,words in ca_conn.execute('select spl_id,words from acronym'):
        set_id = get_set_id(spl_id, ca_conn)
        if set_id==False: set_id = ""
        acronyms_data.append([set_id, words])
    acronyms_df = pd.DataFrame(acronyms_data, columns=['set_id','words'])
    
    never_match_data = []
    for spl_id,code_id in ca_conn.execute('select spl_id,code_id from never_match'):
        set_id = get_set_id(spl_id, ca_conn)
        if set_id==False: set_id = ""
        code = get_code(code_id, ca_conn)
        if code==False: 
            code = ""
            print(code_id)
        never_match_data.append([set_id, code])
    never_match_df = pd.DataFrame(never_match_data, columns=['set_id','code'])
    
    dont_match_data = []
    for spl_id,code_id,words in ca_conn.execute('select spl_id,code_id,words from dont_match'):
        set_id = get_set_id(spl_id, ca_conn)
        if set_id==False: set_id = ""
        code = get_code(code_id, ca_conn)
        if code==False: 
            code = ""
            print(code_id)
        dont_match_data.append([set_id, code, words])
    dont_match_df = pd.DataFrame(dont_match_data, columns=['set_id','code','words'])
    
    writer = pd.ExcelWriter(f"{working_dir}/{filter_rules_fn}", engine='xlsxwriter')
    acronyms_df.to_excel(writer, index=False, sheet_name='acronyms')
    never_match_df.to_excel(writer, index=False, sheet_name='nevers')
    dont_match_df.to_excel(writer, index=False, sheet_name='donts')
    writer.save()




def list_spls(sort_col="id", ascending=False, filter_curated=None, print_results=True, top_N=20, c_lim=80):
    global indi_conn
    global ca_conn
    global zip_metadata
    
    columns = ["id", "set_id", "date", "title", "version", "spl_id", "answers_n"]

    spls = []
    for indi_spl_id,set_id, in indi_conn.execute('select id,set_id from spl'):
        try: 
            m = zip_metadata[set_id]
            date = m['date']
            version = m['spl_version']
            title = m['title']
            if len(title)>c_lim:
                title = title[:c_lim-3] + "..."

        except:
            date = None
            version = -1
            title = None
        spl_id, cache = get_spl_id(set_id, ca_conn, {})
        if spl_id==False: spl_id = -1
        if not spl_id==False:
            sentence_ids = [int(i[0]) for i in ca_conn.execute('select id from sentences where spl_id=?', (spl_id,))]  # get sentence IDs
            if len(sentence_ids)>0:
                if len(sentence_ids)==1: answers_n = len(list(ca_conn.execute(f'select id from answers where sentence_id=?', (sentence_ids[0],))))
                else: answers_n = len(list(ca_conn.execute(f'select id from answers where sentence_id in {repr(tuple(sentence_ids))}')))
            else:
                answers_n = 0
        else:
            answers_n = 0

        spls.append([indi_spl_id, set_id, date, title, version, spl_id, answers_n])


    spl_df = pd.DataFrame(spls, columns=["id", "set_id", "date", "title", "version", "spl_id", "answers_n"])

    # sort
    if isinstance(sort_col, int):
        sort_col = columns[sort_col]
    spl_df.sort_values(sort_col, ascending=ascending)

    # filter curated
    if filter_curated is None:
        filtered_spl_df = spl_df
    else:
        if filter_curated:
            c = spl_df['answers_n']>0
        else:
            c = spl_df['answers_n']==0
        filtered_spl_df = spl_df[c]

    # print top N
    if isinstance(top_N, int): df = filtered_spl_df.head(top_N)
    else: df = filtered_spl_df
        
    if print_results: print(tabulate(df, headers='keys', showindex=False, tablefmt='psql'))
    return df

# select by number
def select_spl(indi_spl_id, overwrite=True, verbose=True):
    global spl_id
    global set_id
    global indi_conn
    global ca_conn
    global zip_metadata
    global curation_answers_fn
    
    set_id = list(indi_conn.execute('select set_id from spl where id=?', (indi_spl_id,)))[0][0]  # set global set_id
    curation_answers_fn = f"{set_id}_answers.xlsx"
    # add spl to ca_conn
    try:
        m = zip_metadata[set_id]
        date = m['date']
        version = m['spl_version']
        title = m['title']
    except:
        date = None
        version = None
        title = None
    try: 
        c = ca_conn.execute('insert into spl(set_id,title,date,version) values (?,?,?,?)', (set_id,title,date,version))
        spl_id = c.lastrowid  # set global spl_id
        if verbose: print(f'SPL {set_id} added to database with ID: {spl_id}')
    except: 
        if overwrite:
            spl_id, cache = get_spl_id(set_id, ca_conn, {})  # set global spl_id
            ca_conn.execute('replace into spl(id,set_id,title,date,version) values (?,?,?,?,?)', (spl_id,set_id,title,date,version))
            if verbose: print(f'SPL {set_id} overwritten, ID: {spl_id}')
        else:
            spl_id, cache = get_spl_id(set_id, ca_conn, {})  # set global spl_id
            if verbose: print(f'SPL {set_id} already in database, ID: {spl_id}')
                
# pick random SPL
def pick_random_spl(filter_curated=False):
    spl_space = list_spls(filter_curated=False, print_results=False, top_N=None)
    if len(spl_space)>0:
        indi_spl_id = int(spl_space.sample(1)['id'])
        select_spl(indi_spl_id)
        return indi_spl_id
    else:
        return None




def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def get_sentence(s_id, cache):
    if not s_id in cache:
        s = list(db_conn.execute(f"select id, onto_id, predicate_type, source, string, sentences, expanded_sentences from strings where id='{s_id}'"))[0]
        cache[s_id] = {}
        cache[s_id]['value'] = {'onto_id': s[1],
                       'predicate_type': s[2],
                       'source': s[3],
                       'string': s[4],
                       'sentence': eval(s[5]),
                       'expanded_sentence': eval(s[6])}
    
    cache[s_id]['timestamp'] = int(datetime.now().strftime('%s'))
    return cache[s_id]['value'], cache


def get_onto_code(s_id):
    onto_i = id_onto_index[s_id]
    onto_id, source = onto_index_r[onto_i]
    return onto_id, source

def trim_cache(cache, lim):
    sorted_cache = sorted(cache.items(), key=lambda x:x[1]['timestamp'], reverse=True)[:lim]
    return {k:v for k,v in sorted_cache}

def query_word(w):
    codes = set()
    if w.lower() in partial_word_index.keys():
        for pw in partial_word_index[w.lower()]:
            l = 'complete' if pw == w.lower() else 'partial'
            codes.update([(s_id,w_id,l) for s_id,w_id in word_index[pw]])
    return codes

def search_for_onto_term(q, score_dict={'doid':5, 'efo':7, 'icd10':5, 'icd11':4, 'mesh':6, 'mondo':6, 'orphanet':5, 'snomed':6}):
    global nlp
    global lemmatizer

    words = [str(w) for w in nlp(q)]  # split q into words
    
    # group matches by code
    matches = {}
    for i,w in enumerate(words):
        w_matches = query_word(w)
        for s_id,w_id,l in w_matches:
            try: matches[s_id][i] = (w_id,l)
            except: matches[s_id] = {i: (w_id,l)}
    
#     sentences = {}
#     for s_id in matches.keys():
#         sentences[s_id], cache = get_sentence(s_id, cache)
#     trim_cache(cache, 10000)
    
    scores = {}
    
    # prioritise based on completeness of phrase
    for s_id, match_path in matches.items():
        matches_words = {v[0] for k,v in match_path.items()}
        p = len(matches_words) / s_len_index[s_id]
        
        scores[s_id] = {'phrase_proportion': p}
        
    # prioritise based on completeness of each word
    for s_id, match_path in matches.items():
        matches_words = {v[0] for k,v in match_path.items() if v[1] == 'complete'}
        p = len(matches_words) / s_len_index[s_id]
        
        scores[s_id]['word_proportion'] = p
        
    # prioritise based on correct order
    for s_id, match_path in matches.items():
        word_ids = [v[0] for k,v in match_path.items()]
        d = levenshteinDistance(word_ids, sorted(word_ids))
        
        l = s_len_index[s_id]
        
        scores[s_id]['correct_order'] = (l-d)/l
    
    # prioritise based on word gaps
    for s_id, match_path in matches.items():
        sorted_vs = sorted([v[0] for k,v in match_path.items()])
        gaps = 0
        for i in range(len(sorted_vs)-1):
            gaps += sorted_vs[i+1]-sorted_vs[i]
        
        l = s_len_index[s_id]
        
        scores[s_id]['gaps'] = (l-gaps)/l
        
    # prioritise based on source
    for s_id, match_path in matches.items():
        onto_id, source = get_onto_code(s_id)
        scores[s_id]['source'] = score_dict[source]
    
    sorted_scores = sorted(scores.items(), key=lambda x:(x[1]['phrase_proportion'], x[1]['word_proportion'], x[1]['gaps'], x[1]['source'], x[1]['correct_order']), reverse=True)
    
    return_scores = []
    for s_id,scores in sorted_scores:
        onto_id, source = get_onto_code(s_id)
        return_scores.append({'matches': matches[s_id], 'onto_id': onto_id, 'source': source})
                           
    return return_scores

def search_for_onto_code(q):
    if q in partial_code_index.keys():
        return sorted(partial_code_index[q], key=lambda x:x[0])
    else:
        return []

def search_term(q, c_lim=60, top_N=20, print_results=True):
    results = []
    unique_onto_ids = set()
    for r in search_for_onto_term(q):
        if r['onto_id'] in unique_onto_ids: continue
        unique_onto_ids.add(r['onto_id'])
        
        try: p_type, name = code_name_index[r['onto_id']]  # get code name
        except:
            p_type = ""
            name = ""
        p_name = f"{name} ({p_type})"
        if len(p_name)>c_lim:
            n_len = c_lim - len(p_type) - 6
            name = name[:n_len] + "..."
            p_name = f"{name} ({p_type})"
            
        results.append([p_name, r['onto_id'], r['source'], repr(r['matches'])])
        if len(results)>=top_N: break
            
    df = pd.DataFrame(results, columns=["Name","Code","Source","Word matches"])
    
    if print_results: print(tabulate(df, headers='keys', showindex=False, tablefmt='psql'))
    return df

def search_code(q, c_lim=60, top_N=20, print_results=True):
    results = []
    unique_codes = set()
    for code,source in search_for_onto_code(q):
        if code in unique_codes: continue
        unique_codes.add(code)
        
        try: p_type, name = code_name_index[r['onto_id']]  # get code name
        except:
            p_type = ""
            name = ""
        p_name = f"{name} ({p_type})"
        if len(p_name)>c_lim:
            n_len = c_lim - len(p_type) - 6
            name = name[:n_len] + "..."
            p_name = f"{name} ({p_type})"
            
        results.append([code, source, p_name])
        if len(results)>=top_N: break
            
    df = pd.DataFrame(results, columns=["Code","Source","Name"])
    
    if print_results: print(tabulate(df, headers='keys', showindex=False, tablefmt='psql'))
    return df

def closest_efo(q, print_results=True):
    global equivalent_entities_groups_index
    global equivalent_entities_groups_index_r
    global disease_hierarchy_distance_index
    global rev_disease_hierarchy_distance_index
    
    sorted_efo_related_codes = gm.find_closest_efo(q, equivalent_entities_groups_index, equivalent_entities_groups_index_r, disease_hierarchy_distance_index, rev_disease_hierarchy_distance_index)
    data = []
    for code,distance in sorted_efo_related_codes:
        try: p_type, name = code_name_index[code]
        except: 
            name = ""
            p_type = ""
        data.append((code,distance,name))
    df = pd.DataFrame(data, columns=["Code","Distance","Name"])
    if print_results: print(tabulate(df, headers='keys', showindex=False, tablefmt='psql'))

    return df




