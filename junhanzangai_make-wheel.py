!pip install zarr
!pip install connected-components-3d
!pip install segmentation-models-pytorch


# 예: zarr, connected-components-3d, segmentation-models-pytorch 패키지 및
#     의존성 패키지 등을 모아서 wheel로 만든 뒤, wheelhouse 폴더에 저장
!pip wheel zarr connected-components-3d segmentation-models-pytorch -w wheelhouse/


!zip -r wheelhouse.zip /kaggle/working/wheelhouse




