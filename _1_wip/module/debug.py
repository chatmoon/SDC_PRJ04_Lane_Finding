from step0 import PARSE_ARGS
import os

def debug_window_centroids_save(file_name='window_centroids.p'):
    file_Name = args.out + file_name
    if len(window_centroids) > 0:
        if os.path.exists(file_Name):
            fileObject = open(file_Name, 'rb')
            a = pickle.load(fileObject)
            a[len(a) + 1] = window_centroids
            fileObject.close()
            fileObject = open(file_Name, 'wb')
            pickle.dump(a, fileObject)
        else:
            fileObject = open(file_Name, 'wb')
            pickle.dump({1: window_centroids}, fileObject)
            fileObject.close()

def debug_window_centroids_plot_(file_name='window_centroids.p'):
    file_Name = args.out + file_name
    fileObject = open(file_Name,'rb') #, encoding="utf8")
    wc0 = pickle.load(fileObject)
    wc1 = np.array([  [ list(tuple0) for tuple0 in list0 ] for list0 in wc0.values() ])

    count = 0
    for frame in wc1:
        plt.plot(frame[:,0],range(1, 10),'r--')
        plt.plot(frame[:,1],range(1, 10),'b--')
        count += 1
    print('count: ', count)