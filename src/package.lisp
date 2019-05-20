(uiop:define-package #:bops
  (:use #:cl :sb-ext :sb-c :iterate)
  (:import-from :sb-sys
                :system-area-pointer)
  (:import-from :sb-assem
                :inst)
  (:import-from :sb-vm
                :unsigned-reg
                :sap-reg
                :unsigned-num
                :single-sse-reg)
  (:import-from :sb-x86-64-asm
                :movaps
                :make-ea
                :divps
                :addps)
  (:import-from :sb-c
                :move)
  (:export #:make-random-bit-vector
           #:split-bitplane
           #:fuse-bitplane-uint8
           #:array-pad
           #:bit-and-vec
           #:sign
           #:test
           #:test2))
