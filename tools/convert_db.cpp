#include <iostream>
#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

using namespace caffe;
using boost::scoped_ptr;

DEFINE_string(backend_in, "leveldb", "The backend_in {leveldb, lmdb}");
DEFINE_string(backend_out, "lmdb", "The backend_out {leveldb, lmdb}");
int main(int argc, char** argv) {
#ifdef GFLAGS_GLFAGS_H_
    namespace gflags = google;
#endif
    gflags::SetUsageMessage("Extract all keys from given db: [FLAGS] INPUT_DB\n");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (argc < 2 || argc > 3) {
      gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/extract_id");
      return 1;
    }
    // Open old db
    scoped_ptr<db::DB> db_in(db::GetDB(FLAGS_backend_in));
    db_in->Open(argv[1], db::READ);
    scoped_ptr<db::Cursor> cursor(db_in->NewCursor());
    // Create new DB
    scoped_ptr<db::DB> db_out(db::GetDB(FLAGS_backend_out));
    db_out->Open(argv[2], db::NEW);
    scoped_ptr<db::Transaction> txn(db_out->NewTransaction());
    // Convert db
    int count = 0;
    while (cursor->valid()) {
      //string id = cursor->key();
      //std::cout << id << std::endl;
      txn->Put(cursor->key(), cursor->value());
    
      if (++count % 1000 == 0) {
        // Commit db
        txn->Commit();
        txn.reset(db_out->NewTransaction());
        LOG(INFO) << "Processed " << count << " items.";
      }
      cursor->Next();
    }
    // write the last batch
    if (count % 1000 != 0) {
      txn->Commit();
      LOG(INFO) << "Processed " << count << " items.";
    }
    db_in->Close();
    return 0;
}