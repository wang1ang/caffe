#include "caffe/util/db.hpp"
#include "caffe/util/db_leveldb.hpp"
#include "caffe/util/db_lmdb.hpp"
#include "caffe/util/db_txt.hpp"

#include <string>

namespace caffe { namespace db {

DB* GetDB(DataParameter::DB backend) {
  switch (backend) {
#ifdef USE_LEVELDB
  case DataParameter_DB_LEVELDB:
    return new LevelDB();
#endif  // USE_LEVELDB
#ifdef USE_LMDB
  case DataParameter_DB_LMDB:
    return new LMDB();
#endif  // USE_LMDB
  case DataParameter_DB_TXTDB:
    return new TxtDB();
  default:
    LOG(FATAL) << "Unknown database backend";
    return NULL;
  }
}

DB* GetDB(const string& backend) {
#ifdef USE_LEVELDB
  if (backend == "leveldb") {
    return new LevelDB();
  }
#endif  // USE_LEVELDB
#ifdef USE_LMDB
  if (backend == "lmdb") {
    return new LMDB();
  }
#endif  // USE_LMDB
  if (backend == "txt") {
    return new TxtDB();
  }
  LOG(FATAL) << "Unknown database backend";
  return NULL;
}

}  // namespace db
}  // namespace caffe
