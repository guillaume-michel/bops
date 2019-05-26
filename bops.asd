(defsystem "bops"
  :description "Binary operations in Common Lisp"
  :version "0.0.1"
  :author "Guillaume MICHEL"
  :mailto "contact@orilla.fr"
  :homepage "http://orilla.fr"
  :license "MIT license (see COPYING)"
  :depends-on ("iterate"
               "array-operations"
               "cl-slice"
               "cl-idx"
               "trivial-garbage")
  :in-order-to ((test-op (test-op "bops-tests")))
  :components ((:static-file "COPYING")
               (:static-file "README.md")
               (:module "src"
                        :serial t
                        :components ((:file "package")
                                     (:file "vops")
                                     (:file "utils")
                                     (:file "array-utils")
                                     (:file "bops")
                                     (:file "ml-functions")
                                     (:file "dense")
                                     (:file "mutation")
                                     (:file "crossover")
                                     (:file "neural-network")
                                     (:file "fully-connected")
                                     (:file "fuse-bitplane")
                                     (:file "softmax")
                                     (:file "sequential")
                                     (:file "mlp")
                                     (:file "mnist")
                                     (:file "train")))))
