# Helper function: command-line / parse parameters 
class PARSE_ARGS(object):
    # TODO: replace it by collections.namedtuple , ref: email PyTricks 30.03.18
    def __init__(self,
                 path,
                 incorner = [9,6],
                 column   = 5,
                 figsize  = (15, 8),
                 ksize    = 3,
                 offset   = 100,
                 nwindows = 9,
                 margin   = 100,
                 minpix   = 50,
                 to_plot  = False):

        self.path = path # root directory path
        self.cali = self.path + 'camera_cal/' # calibration images directory path
        self.out  = self.path + 'output_images/'
        self.test = self.path + 'test_images/' #
        self.file_csv = self.path + 'data/tabular/'
        self.incorner = incorner #
        self.column   = column # 
        self.figsize  = figsize
        self.ksize    = 3
        self.offset   = 100
        self.nwindows = 9    # number of sliding windows
        self.margin   = 100  # width of the windows +/- margin
        self.minpix   = 50   # minimum number of pixels found to recenter window
        self.to_plot  = to_plot

    def path(self):
        return self.path
    def cali(self):
        return self.cali
    def out(self):
        return self.out
    def test(self):
        return self.test
    def file_csv(self):
        return self.file_csv
    def incorner(self):
        return self.incorner
    def column(self):
        return self.column
    def figsize(self):
        return self.figsize
    def ksize(self):
        return self.ksize
    def offset(self):
        return self.offset
    def nwindows(self):
        return self.nwindows
    def margin(self):
        return self.margin
    def minpix(self):
        return self.minpix
    def to_plot(self):
        return self.to_plot


def main():
    # parameters
    directory = 'D:/USER/_PROJECT_/_PRJ04_/_1_WIP/_1_forge/_3_retro/'
    args      = PARSE_ARGS(path=directory)

    # test each args
    print(args.path)
    print(args.cali)
    print(args.out)
    print(args.test)
    print(args.file_csv)
    print(args.incorner)
    print(args.column)
    print(args.figsize)
    print(args.ksize)
    print(args.offset)
    print(args.nwindows)
    print(args.margin)
    print(args.minpix)
    print(args.to_plot)

    '''
    https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
    '''

if __name__ == '__main__':
    main()
