#ifndef CAFFE_UTIL_DB_TXT_HPP
#define CAFFE_UTIL_DB_TXT_HPP

#include <string>
#include "boost/algorithm/string.hpp"

#include "caffe/util/db.hpp"
namespace caffe { namespace db {

class TxtDBCursor : public Cursor {
 public:
  explicit TxtDBCursor(std::fstream *txt)
    : txt_(txt) {
    SeekToFirst();
    CHECK(txt_->good()) << "Text is not opened.";
    Next();
  }
  ~TxtDBCursor() {}
  virtual void SeekToFirst() { txt_->seekg(0, std::ios_base::beg); }
  virtual void Next() { 
    string line;
    while (!std::getline(*txt_, line))
    {
      if (txt_->bad()) {
          break;
        // IO error
      } else if (txt_->eof()) {
          SeekToFirst();
      }
    }
    size_t p = line.find('\t');
    if (p == string::npos)
        // error
    key_ = line.substr(0, p);
    value_ = line.substr(p+1);
  }
  virtual string key() { return key_; }
  virtual string value() {
      std::vector<std::string> feat_string;
      boost::split(feat_string, value_, boost::is_any_of("\t"));
      int dim = feat_string.size();
      caffe::Datum datum;
      datum.set_height(1);
      datum.set_width(dim);
      datum.set_channels(1);
      datum.clear_data();
      datum.clear_float_data();
      for (int i = 0; i < dim; i++)
          datum.add_float_data((float)atof(feat_string[i].c_str()));
      string out;
      CHECK(datum.SerializeToString(&out));
      return out; 
      }
  virtual bool valid() { return txt_->good(); }

 private:
  std::fstream *txt_;
  string key_;
  string value_;
};

class TxtDBTransaction : public Transaction {
 public:
  explicit TxtDBTransaction(std::fstream *txt) : txt_(txt) { CHECK_NOTNULL(txt_); }
  virtual void Put(const string& key, const string& value) {
    caffe::Datum d;
    d.clear_data();
    d.clear_float_data();
    d.ParseFromString(value);
    (*txt_) << key;
    auto size = d.float_data_size(); // d.channels() * d.height() * d.width()
    for (int i = 0; i < size; i++)
      (*txt_) << "\t" << d.float_data(i);
    (*txt_) << std::endl;
  }
  virtual void Commit() {
    txt_->flush();
    CHECK(!txt_->bad()) << "Failed to write batch to text "
                        << std::endl;
  }

 private:
  std::fstream *txt_;
  DISABLE_COPY_AND_ASSIGN(TxtDBTransaction);
};

class TxtDB : public DB {
 public:
  TxtDB() { }
  virtual ~TxtDB() { Close(); }
  virtual void Open(const string& source, Mode mode);
  virtual void Close() {
    if (txt.is_open()) {
      txt.close();
    }
  }
  virtual TxtDBCursor* NewCursor() {
    return new TxtDBCursor(&txt);
  }
  virtual TxtDBTransaction* NewTransaction() {
    return new TxtDBTransaction(&txt);
  }

 private:
  //leveldb::DB* db_;
  std::fstream txt;
};


}  // namespace db
}  // namespace caffe

#endif  // CAFFE_UTIL_DB_TXT_HPP
