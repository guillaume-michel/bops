(in-package :bops)

(defun load-mnist (&optional (directory-path (merge-pathnames "data/mnist/"
                                                              (user-homedir-pathname))))
  "Load MNIST dataset from disk and returns (VALUES x-train y-train x-test y-test) in that order."
  (let ((x-train-path (merge-pathnames "train-images-idx3-ubyte" directory-path))
        (y-train-path (merge-pathnames "train-labels-idx1-ubyte" directory-path))
        (x-test-path (merge-pathnames "t10k-images-idx3-ubyte" directory-path))
        (y-test-path (merge-pathnames "t10k-labels-idx1-ubyte" directory-path)))
    (list
     (idx:read-from-file x-train-path)
     (idx:read-from-file y-train-path)
     (idx:read-from-file x-test-path)
     (idx:read-from-file y-test-path))))

(defun prepare-mnist (datas &key
                              (paddings '((0 0) (2 2) (2 2)))
                              (pad-value 0))
  (flet ((prepare-x (x)
           (let ((batch-size (array-dimension x 0)))
             ;; final slice is to get a simple-array instead of an array
             (cl-slice:slice (aops:reshape (aops:permute '(0 3 1 2)
                                                         (split-bitplane (array-pad x :paddings paddings :pad-value pad-value)))
                                           `(,batch-size 8 t))
                             t t t))))
    (destructuring-bind (x-train y-train x-test y-test) datas
      (list
       (prepare-x x-train)
       y-train
       (prepare-x x-test)
       y-test))))
