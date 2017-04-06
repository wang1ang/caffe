#include "caffe/text_data_transformer.hpp"
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string.hpp>

namespace caffe{

template<class Dtype>
shared_ptr<ITextHashing<Dtype>> ITextHashing<Dtype>::CreateInstanceAndInit(const string& algo_name, const TextHashingParameter& param){
    shared_ptr<ITextHashing<Dtype>> obj;
    if (algo_name == "chargram")
        obj.reset(new ChargramHashing<Dtype>());
    if (algo_name == "bow")
        obj.reset(new BOWHashing<Dtype>());
    if (algo_name == "wordgrambasedchargram")
        obj.reset(new NLettergramHashing<Dtype>());
    CHECK(obj.get() != nullptr) << "non-supported algorithm: " << algo_name;
    obj->Init(param);
    return obj;
}

template<class Dtype>
vector<string> WordHashingBase<Dtype>::Tokenize(string& str,  vector<uint32_t>& ngram) const {
    CHECK(this->param_.has_word_hashing_param()) << "word_hashing_param should be assigned";
    CHECK(ngram.size() > 0) << "ngram should assign at least one value";
    sort(ngram.begin(), ngram.end());
    CHECK(ngram[0] > 0) << "ngram should not be zero";
    // split str using delimiters
    boost::algorithm::trim_right(str);
    boost::to_lower(str);
    tokenizer  tok(str, delimiters_);
    // building ngram
    vector<string> res_ngram, words;
    if (this->param_.word_hashing_param().start_end_symbol())
        words.push_back("<s>");
    for (tokenizer::iterator it = tok.begin(); it != tok.end(); ++it)
        words.push_back(*it);
    if (this->param_.word_hashing_param().start_end_symbol())
        words.push_back("<s>");
    for (uint32_t i = 0; i < words.size(); ++i) {
        string cur_gram("");
        for (uint32_t cur_word_index = i, cur_ngram_index = 0; cur_word_index < words.size() && cur_ngram_index < ngram.size(); cur_word_index++) {
            if (cur_gram.length() == 0)
                cur_gram += words[cur_word_index];
            else
                cur_gram = cur_gram + " " + words[cur_word_index];
            if (cur_word_index - i + 1 == ngram[cur_ngram_index]) {
                if (cur_gram != "<s>")
                    res_ngram.push_back(cur_gram);
                cur_ngram_index++;
            }
        }
    }
    return res_ngram;
}

template<class Dtype>
void WordHashingBase<Dtype>::LoadDictionary(){
    CHECK(this->param_.has_dictionary_file()) << "dictionary file path is not given";
    std::ifstream inf(this->param_.dictionary_file().c_str());
    CHECK(inf) << "can't open dictionary file: " << this->param_.dictionary_file();
    this->dict_.clear();
    string line;
    int fid = 0;
    while (std::getline(inf, line)){
        boost::algorithm::trim_right(line);
        CHECK(this->dict_.find(line) == this->dict_.end())<< "there are duplicate words in the dictionary file, word: " << line;
        this->dict_.insert(std::make_pair(line, fid++));
    }
    LOG(INFO) << "read dictionary file: " << this->param_.dictionary_file() << ". vocabulary size: " << this->dict_.size();
}

template<class Dtype>
void WordHashingBase<Dtype>::BuildDictionary(){
    CHECK(this->param_.has_build_dict_param() && this->param_.build_dict_param().has_train_data() && this->param_.build_dict_param().has_save_path()) << "parameters for building dictionary is missing";
    this->dict_.clear();
    unordered_map<string, float> df;
    unordered_map<string, bool> appeared;
    uint32_t doc_num = 0;
    // Read the dictionary generation source file
    LOG(INFO) << "Opening dictionary training data file " << this->param_.build_dict_param().train_data();
    std::ifstream inf;
    inf.open(this->param_.build_dict_param().train_data());
    CHECK(inf) << "can't open file " << this->param_.build_dict_param().train_data();
    string line;
    while (std::getline(inf, line)){
        doc_num += 1;
        vector <string> ps;
        boost::split(ps, line, boost::is_any_of("\t"));
        CHECK_EQ(ps.size(), 2) << "incorrect line, it must be two columns! It is now: " << line;
        appeared.clear();
        vector<string> ngrams = this->ParseText(ps[0]);
        for (auto it = ngrams.begin(); it != ngrams.end(); ++it) {
            std::string cur_word(it->c_str());
            if (appeared.find(cur_word) == appeared.end()) {
                appeared[cur_word] = true;
                if (df.find(cur_word) != df.end())
                    df[cur_word] += 1;
                else
                    df[cur_word] = 1;
            }
        }
    }
    std::vector<std::pair<std::string, float> > word_score;
    int word_all_num = 0;
    for (auto iter = df.begin(); iter != df.end(); iter++)
        word_score.push_back(std::make_pair(iter->first, iter->second));
    // sort according to df
    sort(word_score.begin(), word_score.end(), [](const std::pair<std::string, float> &left, const std::pair<std::string, float> &right) {return left.second > right.second;});
    // save the result in dict_
    std::ofstream outf(this->param_.build_dict_param().save_path());
    uint32_t dict_size = word_score.size();
    if (this->param_.has_build_dict_param() && this->param_.build_dict_param().has_dict_size())
        dict_size = this->param_.build_dict_param().dict_size();
    this->dict_.clear();
    int fid = 0;
    for (uint32_t i = 0; i < dict_size && i < word_score.size(); i++) {
        string line(word_score[i].first);
        boost::algorithm::trim_right(line);
        CHECK(this->dict_.find(line) == this->dict_.end()) << "there are duplicate words in the dictionary file, word: " << line;
        this->dict_.insert(std::make_pair(line, fid++));
        outf << line << std::endl;
    }
    outf.close();
}

template<class Dtype>
void ChargramHashing<Dtype>::LoadDictionary(){
    CHECK(this->param_.has_dictionary_file()) << "dictionary file path is not given";
    std::ifstream inf(this->param_.dictionary_file());
    CHECK(inf) << "can't open dictionary file: " << this->param_.dictionary_file();
    this->dict_.clear();
    string line;
    int fid = 0;
    while (std::getline(inf, line)){
        boost::algorithm::trim_right(line);
        CHECK(line.length() == 1) << "there are non-character entry in the dictionary file, word: " << line;
        CHECK(this->dict_.find(line[0]) == this->dict_.end()) << "there are duplicate words in the dictionary file, word: " << line;
        this->dict_.insert(std::make_pair(line[0], fid++));
    }
    LOG(INFO) << "read dictionary file: " << this->param_.dictionary_file() << ". vocabulary size: " << this->dict_.size();
}

template<class Dtype>
void ChargramHashing<Dtype>::BuildDictionary(){
    CHECK(this->param_.has_build_dict_param() && this->param_.build_dict_param().has_train_data() && this->param_.build_dict_param().has_save_path());
    this->dict_.clear();
    // Read the dictionary generation source file
    LOG(INFO) << "Opening dictionary training data file " << this->param_.build_dict_param().train_data();
    std::ifstream inf;
    inf.open(this->param_.build_dict_param().train_data().c_str());
    CHECK(inf) << "can't open file.";
    string line;
    while (std::getline(inf, line)){
        vector <string> ps;
        boost::split(ps, line, boost::is_any_of("\t"));
        CHECK_EQ(ps.size(), 2) << "incorrect line, it must be two columns! It is now: " << line;
        std::string text = ps[0];
        uint32_t max_num_words = std::min((uint32_t)text.length(), this->param_.max_num_words_per_doc());
        for (uint32_t wid = 0; wid < max_num_words; ++wid){
            if (text[wid] == ' ' || text[wid] == '\t')
                continue;
            auto itr = this->dict_.find(text[wid]);
            if (itr == this->dict_.end()){
                dict_[text[wid]] = 1;
            }
        }
    }
    std::ofstream outf(this->param_.build_dict_param().train_data().c_str());
    for (auto iter = this->dict_.begin(); iter != this->dict_.end(); iter++)
        outf << iter->first << std::endl;
    outf.close();
}

template<class Dtype>
void ChargramHashing<Dtype>::Init(const TextHashingParameter& param){
    TextHashingBase<Dtype>::Init(param);
}

template<class Dtype>
void ChargramHashing<Dtype>::Extract(const vector<string> &data, SparseBlob<Dtype>& out_blob) const{
    out_blob.Reshape((int)data.size(), this->param_.max_num_words_per_doc(), this->param_.max_num_non_zeros_per_word(), this->dict_.size());
    SparseItem<Dtype>* p_data = out_blob.mutable_cpu_data();
    int element_id = 0;
    for (int sample_id = 0; sample_id < data.size(); ++sample_id){
        const string& text = data[sample_id];
        uint32_t max_num_words = std::min((uint32_t)text.length(), this->param_.max_num_words_per_doc());
        for (uint32_t wid = 0; wid < max_num_words; ++wid){
            auto itr = this->dict_.find(text[wid]);
            if (itr != this->dict_.end()){
                int fid = itr->second;
                p_data[element_id++] = SparseItem<Dtype>(sample_id, wid, fid, Dtype(1.0));
            }            
        }
    }
    if (element_id<out_blob.count())
        p_data[element_id] = SparseItem<Dtype>::NullItem();
}

template<class Dtype>
void BOWHashing<Dtype>::Init(const TextHashingParameter& param) {
    if (param.has_word_hashing_param() && param.word_hashing_param().has_delimiters())
        this->delimiters_ = boost::char_separator<char>{param.word_hashing_param().delimiters().c_str()};
    WordHashingBase<Dtype>::Init(param);
    CHECK(!param.has_max_num_non_zeros_per_word()) << "you shouldn't set max_num_non_zeros_per_word";
}

template<class Dtype>
vector<string> BOWHashing<Dtype>::ParseText(string& str) const {
    CHECK(this->param_.has_word_hashing_param()) << "please clarify the word hashing parameters in configuration prototxt file";
    vector<uint32_t> ngram_vec;
    for (uint32_t i = 1; i <= this->param_.word_hashing_param().word_gram_length(); i++)
        ngram_vec.push_back(i);
    return this->Tokenize(str, ngram_vec);
}

template<class Dtype>
void BOWHashing<Dtype>::Extract(const vector<string> &data, SparseBlob<Dtype>& out_blob) const {
    CHECK(this->param_.has_word_hashing_param()) << "please clarify the bow parameters in configuration prototxt file";
    out_blob.Reshape((int)data.size(), 1, this->param_.max_num_words_per_doc(), this->dict_.size());
    SparseItem<Dtype>* p_data = out_blob.mutable_cpu_data();
    int element_id = 0;
    unordered_map<int, int> term_freq;
    for (int sample_id = 0; sample_id < data.size(); ++sample_id){
        std::string text(data[sample_id].c_str());
        uint32_t max_num_words = std::min((uint32_t)text.length(), this->param_.max_num_words_per_doc());
        vector<string> ngrams = ParseText(text);
        term_freq.clear();
        uint32_t cur_words = 0;
        for (auto it = ngrams.begin(); it != ngrams.end() && cur_words < max_num_words; ++it, ++cur_words) {
            std::string cur_word(it->c_str());
            auto itr = this->dict_.find(cur_word);
            if (itr != this->dict_.end()) {
                int fid = itr->second;
                if (term_freq.find(fid) == term_freq.end()) term_freq[fid] = 1;
                else term_freq[fid] += 1;
            }
        }
        for (auto itr = term_freq.begin(); itr != term_freq.end(); ++itr) {
            p_data[element_id++] = SparseItem<Dtype>(sample_id, 0, itr->first, Dtype(itr->second));
        }
    }
    if (element_id<out_blob.count())
        p_data[element_id] = SparseItem<Dtype>::NullItem();
}

template<class Dtype>
void NLettergramHashing<Dtype>::Init(const TextHashingParameter& param) {
    if (param.has_word_hashing_param() && param.word_hashing_param().has_delimiters())
        this->delimiters_ = boost::char_separator<char>{param.word_hashing_param().delimiters().c_str()};
    WordHashingBase<Dtype>::Init(param);
}

template<class Dtype>
vector<string> NLettergramHashing<Dtype>::TokenToCharNGram(string& token) const {
    CHECK(this->param_.has_word_hashing_param() && this->param_.word_hashing_param().has_nletter_gram_param()) << "nletter_gram_param should be assigned";
    uint32_t char_gram_length = this->param_.word_hashing_param().nletter_gram_param().char_gram_length();
    CHECK(char_gram_length >= 1) << "char_gram_length must be greater than zero, otherwise please use chargram hash method";
    vector<string> char_n_grams;
    string cur_ngram("#");
    for (uint32_t i = 0; i < token.length(); i++) {
        cur_ngram += token[i];
        if (cur_ngram.length() == char_gram_length) {
            char_n_grams.push_back(cur_ngram);
            cur_ngram = cur_ngram.substr(1, cur_ngram.length() - 1);
        }
    }
    // pad if not length is not enough
    for (uint32_t i = cur_ngram.length(); i < char_gram_length; i++)
        cur_ngram += "#";
    char_n_grams.push_back(cur_ngram);
    return char_n_grams;
}

template<class Dtype>
vector<string> NLettergramHashing<Dtype>::ParseText(string& str) const {
    CHECK(this->param_.has_word_hashing_param() && this->param_.word_hashing_param().has_nletter_gram_param()) << "nletter_gram_param should be assigned";
    vector<string> words, ngrams;
    vector<uint32_t> ngram_vec(1, 1);
    words = this->Tokenize(str, ngram_vec);
    for (uint32_t i = 0; i < words.size(); i++) {
        vector<string> char_n_grams = TokenToCharNGram(words[i]);
        for (auto iter = char_n_grams.begin(); iter != char_n_grams.end(); ++iter)
            ngrams.push_back(*iter);
    }
    return ngrams;
}

template<class Dtype>
void NLettergramHashing<Dtype>::Extract(const vector<string> &data, SparseBlob<Dtype>& out_blob) const {
    CHECK(this->param_.has_word_hashing_param() && this->param_.word_hashing_param().has_nletter_gram_param()) << "nletter_gram_param should be assigned";
    out_blob.Reshape((int)data.size(), this->param_.max_num_words_per_doc(), this->param_.max_num_non_zeros_per_word() * this->param_.word_hashing_param().word_gram_length(), this->dict_.size() * this->param_.word_hashing_param().word_gram_length());
    SparseItem<Dtype>* p_data = out_blob.mutable_cpu_data();
    int element_id = 0;
    unordered_map<int, int> term_freq;
    for (int sample_id = 0; sample_id < data.size(); ++sample_id){
        std::string text(data[sample_id].c_str());
        uint32_t max_num_wordgrams = std::min((uint32_t)text.length(), this->param_.max_num_words_per_doc());
        uint32_t cur_wordgrams = 0;
        uint32_t max_num_chargrams = std::min((uint32_t)text.length(), this->param_.max_num_non_zeros_per_word());
        uint32_t cur_chargrams = 0;
        vector<uint32_t> ngram_vec(1, this->param_.word_hashing_param().word_gram_length());
        vector<string> word_n_grams = this->Tokenize(text, ngram_vec);
        for (uint32_t i = 0; i < word_n_grams.size() && i < this->param_.max_num_words_per_doc(); i++) {
            vector<string> words;
            boost::split(words, word_n_grams[i], boost::is_any_of(" "));
            for (uint32_t j = 0; j < words.size(); j++) {
                term_freq.clear();
                vector<string> chargrams = TokenToCharNGram(words[j]);
                for (uint32_t k = 0; k < chargrams.size() && k < this->param_.max_num_non_zeros_per_word(); k++) {
                    auto itr = this->dict_.find(chargrams[k]);
                    if (itr != this->dict_.end()) {
                        int fid = itr->second;
                        if (term_freq.find(fid) == term_freq.end()) term_freq[fid] = 1;
                        else term_freq[fid] += 1;
                    }
                }
                for (auto itr = term_freq.begin(); itr != term_freq.end(); ++itr) {
                    p_data[element_id++] = SparseItem<Dtype>(sample_id, i, j * this->dict_.size() + itr->first, Dtype(itr->second));
                }
            }
        }
    }
    if (element_id<out_blob.count())
        p_data[element_id] = SparseItem<Dtype>::NullItem();
}


INSTANTIATE_CLASS(ITextHashing);
INSTANTIATE_CLASS(WordHashingBase);
INSTANTIATE_CLASS(ChargramHashing);
INSTANTIATE_CLASS(BOWHashing);
INSTANTIATE_CLASS(NLettergramHashing);

}