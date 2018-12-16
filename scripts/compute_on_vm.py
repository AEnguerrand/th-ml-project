from download import test as dl_test, train as dl_train
from load import test as l_test, train as l_train
from scripts import process_test, process_train
import sys
from multiprocessing import Pool



def compute_on_vm():
    if len(sys.argv) >= 2 and sys.argv[1] == 'nodownload':
        print("[COMPUTE] [NO Download]")
    else:
        print("[COMPUTE] [Download train dataset]")
        dl_train.download_train()
        print("[COMPUTE] [Download test dataset]")
        dl_test.download_test()
    print("[COMPUTE] [Load metadata (Train and Test]")
    th_pool = Pool()
    th_pool.apply_async(l_train.load_metadata())
    th_pool.apply_async(l_test.load_metadata())
    th_pool.close()
    th_pool.join()
    print("[COMPUTE] [Load train set]")
    l_train.load_set()
    print("[COMPUTE] [Generate test set]")
    if len(sys.argv) >= 3 and sys.argv[2] == 'notesttrans':
        print("[COMPUTE] [NO TEST TRANSFORM to CHUNK]")
    else:
        l_test.convert_test_set_to_chunk()
    print("[COMPUTE] [Load metadata]")
    print("[COMPUTE] [Process train dataset]")
    process_train.process_train()
    print("[COMPUTE] [Process test dataset]")
    process_test.process_test_multithread()
    print("[COMPUTE] [Finish]")


if __name__ == '__main__':
    compute_on_vm()