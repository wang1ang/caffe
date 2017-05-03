#include <iostream>
#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

using namespace caffe;
using boost::scoped_ptr;

DEFINE_string(backend, "leveldb", "The backend {leveldb, lmdb}");
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
    scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
    db->Open(argv[1], db::READ);
    scoped_ptr<db::Cursor> cursor(db->NewCursor());

    while (cursor->valid()) {
        string id = cursor->key();
        std::cout << id << std::endl;
        cursor->Next();
    }
    db->Close();
}