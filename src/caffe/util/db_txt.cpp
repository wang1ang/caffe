
#include "caffe/util/db_txt.hpp"

#include <string>

namespace caffe { namespace db {

void TxtDB::Open(const string& source, Mode mode) {
  if (mode == READ)
	txt.open(source, std::fstream::in);
  if (mode == NEW)
    txt.open(source, std::fstream::out);
  CHECK(txt.good()) << "Failed to open text file " << source << std::endl;
  LOG(INFO) << "Opened text file " << source;
}

}  // namespace db
}  // namespace caffe
