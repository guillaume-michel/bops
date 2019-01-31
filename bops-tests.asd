(defsystem "bops-tests"
  :description "bops unit tests"
  :author "Guillaume MICHEL"
  :mailto "contact@orilla.fr"
  :license "MIT license (see COPYING)"
  :depends-on ("bops"
               "fiveam")
  :perform (test-op (o s) (uiop:symbol-call :bops-tests :run-tests))
  :components ((:module "t"
                :serial t
                :components ((:file "package")
                             (:file "tests")))))
