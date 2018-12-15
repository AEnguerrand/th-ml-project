from download import test as dl_test, train as dl_train
from scripts import process_test_multithreaded, process_train
import sys


def compute_on_azure():
    if len(sys.argv) >= 2 and sys.argv[1] == 'nodownload':
        print("[COMPUTE] [NO Download]")
    else:
        print("[COMPUTE] [Download train dataset]")
        dl_train.download_train()
        print("[COMPUTE] [Download test dataset]")
        dl_test.download_test()
    print("[COMPUTE] [Process train dataset]")
    process_train.process_train()
    print("[COMPUTE] [Process test dataset]")
    process_test_multithreaded.process_test_multithreaded()
    print("[COMPUTE] [Finish]")


if __name__ == '__main__':
    compute_on_azure()