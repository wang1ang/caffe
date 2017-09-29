#include <iostream>
#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

using namespace caffe;
using boost::scoped_ptr;

DEFINE_string(backend_in, "leveldb", "The backend_in {leveldb, lmdb}");
DEFINE_string(backend_out, "lmdb", "The backend_out {leveldb, lmdb}");
DEFINE_string(dup, "6", "# dups of second set");
int main(int argc, char** argv) {
#ifdef GFLAGS_GLFAGS_H_
    namespace gflags = google;
#endif
    gflags::SetUsageMessage("Extract all keys from given db: [FLAGS] INPUT_DB\n");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (argc < 6) {
        gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/merge_db");
        return 1;
    }
    LOG(INFO) << "main: " << argv[1];
    LOG(INFO) << "list: " << argv[2];
    LOG(INFO) << "delta: " << argv[3];
    LOG(INFO) << "list: " << argv[4];
    LOG(INFO) << "target: " << argv[5];
    LOG(INFO) << "list: " << argv[6];
    
    // load list file
    std::vector<string> lines_main;
    std::vector<string> lines_delta;
    std::string line;
    std::ifstream infile(argv[2]);
    while (std::getline(infile, line)) {
        lines_main.push_back(line);
    }
    infile.close();
    infile.open(argv[4]);
    while (std::getline(infile, line)) {
        lines_delta.push_back(line);
    }
    infile.close();
    
    // Open db
    scoped_ptr<db::DB> db_main(db::GetDB("lmdb"));
    db_main->Open(argv[1], db::READ);
    scoped_ptr<db::Cursor> cursor_main(db_main->NewCursor());
    scoped_ptr<db::DB> db_delta(db::GetDB("lmdb"));
    db_delta->Open(argv[3], db::READ);
    scoped_ptr<db::Cursor> cursor_delta(db_delta->NewCursor());
	// Create new DB
    scoped_ptr<db::DB> db_out(db::GetDB("lmdb"));
    db_out->Open(argv[5], db::NEW);
    scoped_ptr<db::Transaction> txn(db_out->NewTransaction());

    int dup = atoi(FLAGS_dup.c_str());
	LOG(INFO) << "dup: " << dup;
    int n_main = lines_main.size();
    int n_delta = lines_delta.size();
    std::ofstream outfile(argv[6]);
	long long i = 0, j = 0;
    while (i < n_main || j < n_delta * dup) {
        if (i * n_delta * dup <= j * n_main) {
			outfile << lines_main[i] << std::endl;
			txn->Put(cursor_main->key(), cursor_main->value());
			//outfile << "i: " << i << std::endl;
			i++;
			cursor_main->Next();
		} else {
			outfile << lines_delta[j % n_delta] << std::endl;
			txn->Put(cursor_delta->key(), cursor_delta->value());
			//outfile << "j: " << j << std::endl;
			j++;
			cursor_delta->Next();
		}
		if ((i + j) % 1000 == 0) {
			// Commit db
			txn->Commit();
			txn.reset(db_out->NewTransaction());
			LOG(INFO) << "Processed " << i << ":" << j << " items.";
		}
    }
	outfile.close();
	if ((i + j) % 1000 != 0) {
      txn->Commit();
      LOG(INFO) << "Processed " << i << ":" << j << " items.";
    }
	LOG(INFO) << "main : " << i << "/" << n_main << " copied";
	LOG(INFO) << "delta: " << j << "/" << n_delta << " copied";
    return 0;
}