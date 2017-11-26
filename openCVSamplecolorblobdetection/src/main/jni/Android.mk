LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

export OPENCV_ANDROID_SDK=/Users/dan.nickchen/Documents/OpenCV-android-sdk

OPENCV_INSTALL_MODULES:=on
#OPENCV_CAMERA_MODULES:=on
#OPENCV_LIB_TYPE:=SHARED
ifdef OPENCV_ANDROID_SDK
  ifneq ("","$(wildcard $(OPENCV_ANDROID_SDK)/OpenCV.mk)")
    include ${OPENCV_ANDROID_SDK}/OpenCV.mk
  else
    include ${OPENCV_ANDROID_SDK}/sdk/native/jni/OpenCV.mk
  endif
endif

LOCAL_SRC_FILES  :=
LOCAL_C_INCLUDES += $(LOCAL_PATH)
LOCAL_LDLIBS     += -llog -ldl

LOCAL_MODULE     := color-blob

include $(BUILD_SHARED_LIBRARY)
