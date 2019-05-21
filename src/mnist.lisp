(in-package :bops)

(defun load-mnist (&optional (directory-path (merge-pathnames "data/mnist/"
                                                              (user-homedir-pathname))))
  "Load MNIST dataset from disk and returns (VALUES x-train y-train x-test y-test) in that order."
  (let ((x-train-path (merge-pathnames "train-images-idx3-ubyte" directory-path))
        (y-train-path (merge-pathnames "train-labels-idx1-ubyte" directory-path))
        (x-test-path (merge-pathnames "t10k-images-idx3-ubyte" directory-path))
        (y-test-path (merge-pathnames "t10k-labels-idx1-ubyte" directory-path)))
    (values
     (idx:read-from-file x-train-path)
     (idx:read-from-file y-train-path)
     (idx:read-from-file x-test-path)
     (idx:read-from-file y-test-path))))

(defun prepare-mnist-x (x paddings pad-value)
  (let ((batch-size (array-dimension x 0)))
    (aops:reshape (aops:permute '(0 3 1 2)
                                (split-bitplane (array-pad x :paddings paddings :pad-value pad-value)))
                  `(,batch-size 8 t))))

(defun prepare-mnist (datas &key
                              (paddings '((0 0) (2 2) (2 2)))
                              (pad-value 0))
    (multiple-value-bind (x-train y-train x-test y-test) datas
      (values
       (prepare-mnist-x x-train paddings pad-value)
       y-train
       (prepare-mnist-x x-test paddings pad-value)
       y-test)))

(defun prepare-mnist2 (datas &key
                               (paddings '((0 0) (2 2) (2 2)))
                               (pad-value 0))
  (multiple-value-bind (x-train y-train x-test y-test) datas
    (let ((prepared-x-train nil)
          (prepared-x-test nil))
      (setf prepared-x-train (prepare-mnist-x x-train paddings pad-value))
      (trivial-garbage:gc :full t)
      (setf prepared-x-test (prepare-mnist-x x-test paddings pad-value))
      (trivial-garbage:gc :full t)
      (values prepared-x-train y-train prepared-x-test y-test))))
