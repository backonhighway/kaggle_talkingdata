from multiprocessing.dummy import Pool as ThreadPool


pool = ThreadPool(4)
#results = pool.map(urllib2.urlopen, urls)
pool.close()
pool.join()

