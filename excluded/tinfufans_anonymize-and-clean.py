!pip install openpyxl pydicom


import os
import pydicom
from pydicom.dataset import Dataset

def anonymize_dicom(input_file, output_file=None, overwrite=False):
    
    try:
        # 读取DICOM文件
        ds = pydicom.dcmread(input_file)
        
        # 定义需要脱敏的DICOM标签映射
        tags_to_anonymize = {
            # 患者信息
            "PatientName": "Anonymous",
            "PatientID": "Anonymous",
            "PatientBirthDate": "Anonymous",
            "PatientBirthTime": "Anonymous",
            "OtherPatientIDs": "Anonymous",
            "OtherPatientNames": "Anonymous",
            "PatientBirthName": "Anonymous",
            "PatientSize": "Anonymous",
            "PatientWeight": "Anonymous",
            "PatientAddress": "Anonymous",
            "PatientTelephoneNumbers": "Anonymous",
            
            # 医疗机构和医生信息
            "InstitutionName": "Anonymous",
            "InstitutionAddress": "Anonymous",
            "ReferringPhysicianName": "Anonymous",
            "ReferringPhysicianAddress": "Anonymous",
            "ReferringPhysicianTelephoneNumbers": "Anonymous",
            "InstitutionalDepartmentName": "Anonymous",
            "PerformingPhysicianName": "Anonymous",
            "OperatorsName": "Anonymous",
        }
        
        # 对每个标签进行脱敏处理
        for tag, value in tags_to_anonymize.items():
            if tag in ds:
                setattr(ds, tag, value)
                #print(f"已脱敏: {tag}")
        
        # 特殊处理：Other Patient IDs Sequence (0010,1002)
        if "OtherPatientIDsSequence" in ds:
            # 创建一个新的空序列
            new_sequence = pydicom.sequence.Sequence()
            
            # 创建一个新的数据集
            item = Dataset()
            item.PatientID = "Anonymous"  # 设置序列中可能包含的患者ID字段
            
            # 将数据集添加到序列中
            new_sequence.append(item)
            
            # 替换原有序列
            ds.OtherPatientIDsSequence = new_sequence
            #print("已脱敏: OtherPatientIDsSequence")
        
        # 保存处理后的文件
        if output_file:
            ds.save_as(output_file)
        elif overwrite and False:
            ds.save_as(input_file)
        else:
            # 如果没有提供输出路径且不允许覆盖，则在原文件名后添加"_anon"
            base, ext = os.path.splitext(input_file)
            output_file = f"{base}_anon{ext}"
            ds.save_as(output_file)
        
        #print(f"脱敏完成，文件已保存至: {output_file if output_file else input_file}")
        return True
    except Exception as e:
        print(f"处理文件时出错: {input_file}")
        print(f"错误信息: {str(e)}")
        return False



outputpath0 = "/kaggle/working/rsna-miccai-brain-tumor-radiogenomic-classification/trainanonymized"
filepath0 = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train"
for n,filepath1 in enumerate(os.listdir(filepath0)):
    if n>=2:break
    for filepath2 in os.listdir(os.path.join(filepath0,filepath1)):
        for filepath3 in os.listdir(os.path.join(filepath0,filepath1,filepath2)):
            output_path = os.path.join(outputpath0,filepath1,filepath2,filepath3)
            input_path = os.path.join(filepath0,filepath1,filepath2,filepath3)
            mkdir = os.path.join(outputpath0,filepath1,filepath2)
            os.makedirs(mkdir,exist_ok=True)
            r = anonymize_dicom(input_path, output_path, False)    
            print(mkdir,r)



testcase = '/kaggle/working/rsna-miccai-brain-tumor-radiogenomic-classification/trainanonymized/00058/FLAIR/Image-168.dcm'


ds = pydicom.dcmread(testcase)


ds


import os
import pydicom
import shutil
import pandas as pd
from pydicom.dataset import Dataset
from typing import List, Dict, Set, Tuple, Optional, Any
from openpyxl.styles import PatternFill, Font


class DicomDataCleaner:
    def __init__(self, input_dir: str, output_dir: str):
        """初始化DICOM数据清洗器"""
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.allowed_magnetic_fields = {"1.5", "3.0"}  # 允许的磁场强度(T)
        self.max_slice_thickness = 5.0  # 最大允许层厚(mm)
        self.min_b_value = 1400  # 最小B值
        self.processed_study_uids = set()  # 存储已处理的Study UID
        self.cleaning_details = {}  # 存储详细清理结果（按条件记录）
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    # 以下为原有工具方法（保持不变）
    def _is_dicom_readable(self, file_path: str) -> bool:
        try:
            pydicom.dcmread(file_path)
            return True
        except Exception:
            return False
    
    def _is_mr_data(self, ds: Dataset) -> bool:
        return ds.Modality.lower() == "mr" if "Modality" in ds else False
    
    def _check_unique_study_uid(self, study_uid: str) -> bool:
        return study_uid not in self.processed_study_uids
    
    def _has_required_tags(self, ds: Dataset) -> bool:
        required_tags = [
            "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",
            "Modality", "PatientID", "StudyDate", "SeriesDescription"
        ]
        for tag in required_tags:
            if tag not in ds:
                return False
        return True
    
    def _classify_series(self, ds: Dataset) -> Optional[str]:
        series_desc = ds.get("SeriesDescription", "").lower()
        
        if "dwi" in series_desc or "diffusion" in series_desc:
            if "adc" in series_desc:
                return "adc"
            return "dwi"
        
        if "t2" in series_desc and self._is_traverse_plane(ds):
            tr = self._get_float_value(ds, "RepetitionTime")
            te = self._get_float_value(ds, "EchoTime")
            if tr is not None and te is not None and tr > 2000 and te > 75:
                return "t2_tra"
        return None
    
    def _get_float_value(self, ds: Dataset, tag_name: str) -> Optional[float]:
        value = ds.get(tag_name)
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _is_traverse_plane(self, ds: Dataset) -> bool:
        if "ImageOrientationPatient" in ds:
            orientation = ds.ImageOrientationPatient
            zx, zy, zz = orientation[3], orientation[4], orientation[5]
            return abs(zz) > 0.9
        return False
    
    def _get_b_value(self, ds: Dataset) -> Optional[int]:
        if "DiffusionBValue" in ds:
            try:
                return int(ds.DiffusionBValue)
            except (ValueError, TypeError):
                pass
        series_desc = ds.get("SeriesDescription", "").lower()
        for part in series_desc.split():
            if "b" in part and part.replace("b", "").isdigit():
                return int(part.replace("b", ""))
        return None
    
    def _check_series_integrity(self, series_files: List[str]) -> bool:
        sop_uids = set()
        instance_numbers = set()
        for file in series_files:
            try:
                ds = pydicom.dcmread(file)
                if ds.SOPInstanceUID in sop_uids:
                    return False
                sop_uids.add(ds.SOPInstanceUID)
                if "InstanceNumber" in ds:
                    instance_numbers.add(ds.InstanceNumber)
            except Exception:
                return False
        if instance_numbers:
            min_num = min(instance_numbers)
            max_num = max(instance_numbers)
            if len(instance_numbers) != (max_num - min_num + 1):
                return False
        return True
    
    def _get_magnetic_field_strength(self, ds: Dataset) -> Optional[str]:
        if "MagneticFieldStrength" in ds:
            return str(ds.MagneticFieldStrength)
        return None
    
    def _get_slice_thickness(self, ds: Dataset) -> Optional[float]:
        return self._get_float_value(ds, "SliceThickness")
    
    def process_study(self, study_dir: str) -> Tuple[bool, Dict[str, Any]]:
        """处理单个Study，返回(是否有效, 详细条件检查结果)"""
        study_name = os.path.basename(study_dir)
        # 初始化详细检查结果（所有条件默认False）
        check_results = {
            "有可读DICOM文件": False,
            "是MR数据": False,
            "StudyUID唯一": False,
            "包含T2横断面序列": False,
            "包含ADC序列": False,
            "包含DWI序列": False,
            "T2序列完整": False,
            "ADC序列完整": False,
            "DWI序列完整": False,
            "DWI包含有效B值": False,
            "T2磁场强度合规": False,
            "ADC磁场强度合规": False,
            "DWI磁场强度合规": False,
            "T2层厚合规": False,
            "ADC层厚合规": False,
            "DWI层厚合规": False,
            "总体有效": False,
            "失败原因": []
        }
        
        # 1. 检查是否有可读DICOM文件
        dicom_files = []
        for root, _, files in os.walk(study_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if self._is_dicom_readable(file_path):
                    dicom_files.append(file_path)
        if not dicom_files:
            check_results["失败原因"].append("Study目录中没有可读的DICOM文件")
            self.cleaning_details[study_name] = check_results
            return False, check_results
        check_results["有可读DICOM文件"] = True
        
        # 2. 检查是否为MR数据
        first_ds = pydicom.dcmread(dicom_files[0])
        if not self._is_mr_data(first_ds):
            check_results["失败原因"].append("非MR数据")
            self.cleaning_details[study_name] = check_results
            return False, check_results
        check_results["是MR数据"] = True
        
        # 3. 检查StudyUID唯一性
        study_uid = first_ds.StudyInstanceUID
        if not self._check_unique_study_uid(study_uid):
            check_results["失败原因"].append(f"重复的StudyInstanceUID: {study_uid}")
            self.cleaning_details[study_name] = check_results
            return False, check_results
        check_results["StudyUID唯一"] = True
        self.processed_study_uids.add(study_uid)
        
        # 4. 按Series分组并分类
        series_groups = {}
        for file in dicom_files:
            ds = pydicom.dcmread(file)
            series_uid = ds.SeriesInstanceUID
            if series_uid not in series_groups:
                series_groups[series_uid] = []
            series_groups[series_uid].append(file)
        
        t2_tra_series = []
        adc_series = []
        dwi_series = []
        for series_uid, files in series_groups.items():
            ds = pydicom.dcmread(files[0])
            series_type = self._classify_series(ds)
            if series_type == "t2_tra":
                t2_tra_series.append((series_uid, files))
            elif series_type == "adc":
                adc_series.append((series_uid, files))
            elif series_type == "dwi":
                dwi_series.append((series_uid, files))
        
        # 5. 检查是否包含必要序列
        check_results["包含T2横断面序列"] = len(t2_tra_series) > 0
        check_results["包含ADC序列"] = len(adc_series) > 0
        check_results["包含DWI序列"] = len(dwi_series) > 0
        if not all([check_results["包含T2横断面序列"], 
                   check_results["包含ADC序列"], 
                   check_results["包含DWI序列"]]):
            check_results["失败原因"].append("缺少必要序列（T2/ADC/DWI）")
            self.cleaning_details[study_name] = check_results
            return False, check_results
        
        # 6. 检查序列完整性
        valid_t2 = [f for s, f in t2_tra_series if self._check_series_integrity(f)]
        valid_adc = [f for s, f in adc_series if self._check_series_integrity(f)]
        valid_dwi = [f for s, f in dwi_series if self._check_series_integrity(f)]
        
        check_results["T2序列完整"] = len(valid_t2) > 0
        check_results["ADC序列完整"] = len(valid_adc) > 0
        check_results["DWI序列完整"] = len(valid_dwi) > 0
        if not all([check_results["T2序列完整"], 
                   check_results["ADC序列完整"], 
                   check_results["DWI序列完整"]]):
            check_results["失败原因"].append("存在不完整的序列")
            self.cleaning_details[study_name] = check_results
            return False, check_results
        
        # 7. 检查DWI的B值
        valid_dwi_series = []
        for series_uid, files in valid_dwi:
            has_valid_b = any(self._get_b_value(pydicom.dcmread(f)) >= self.min_b_value 
                             for f in files if self._get_b_value(pydicom.dcmread(f)) is not None)
            if has_valid_b:
                valid_dwi_series.append((series_uid, files))
        check_results["DWI包含有效B值"] = len(valid_dwi_series) > 0
        if not check_results["DWI包含有效B值"]:
            check_results["失败原因"].append(f"DWI缺少b值>={self.min_b_value}的图像")
            self.cleaning_details[study_name] = check_results
            return False, check_results
        
        # 8. 检查磁场强度
        t2_field_ok = all(self._get_magnetic_field_strength(pydicom.dcmread(f[0])) in self.allowed_magnetic_fields 
                         for f in valid_t2)
        adc_field_ok = all(self._get_magnetic_field_strength(pydicom.dcmread(f[0])) in self.allowed_magnetic_fields 
                         for f in valid_adc)
        dwi_field_ok = all(self._get_magnetic_field_strength(pydicom.dcmread(f[0])) in self.allowed_magnetic_fields 
                         for f in valid_dwi_series)
        
        check_results["T2磁场强度合规"] = t2_field_ok
        check_results["ADC磁场强度合规"] = adc_field_ok
        check_results["DWI磁场强度合规"] = dwi_field_ok
        if not all([t2_field_ok, adc_field_ok, dwi_field_ok]):
            check_results["失败原因"].append("磁场强度不符合要求（需1.5T或3.0T）")
            self.cleaning_details[study_name] = check_results
            return False, check_results
        
        # 9. 检查层厚
        t2_thickness_ok = all(self._get_slice_thickness(pydicom.dcmread(f[0])) is not None 
                            and self._get_slice_thickness(pydicom.dcmread(f[0])) <= self.max_slice_thickness 
                            for f in valid_t2)
        adc_thickness_ok = all(self._get_slice_thickness(pydicom.dcmread(f[0])) is not None 
                             and self._get_slice_thickness(pydicom.dcmread(f[0])) <= self.max_slice_thickness 
                             for f in valid_adc)
        dwi_thickness_ok = all(self._get_slice_thickness(pydicom.dcmread(f[0])) is not None 
                             and self._get_slice_thickness(pydicom.dcmread(f[0])) <= self.max_slice_thickness 
                             for f in valid_dwi_series)
        
        check_results["T2层厚合规"] = t2_thickness_ok
        check_results["ADC层厚合规"] = adc_thickness_ok
        check_results["DWI层厚合规"] = dwi_thickness_ok
        if not all([t2_thickness_ok, adc_thickness_ok, dwi_thickness_ok]):
            check_results["失败原因"].append(f"层厚超过限制（最大{self.max_slice_thickness}mm）")
            self.cleaning_details[study_name] = check_results
            return False, check_results
        
        # 所有检查通过
        check_results["总体有效"] = True
        check_results["失败原因"] = ["无"]
        self._save_valid_series(study_uid, valid_t2, valid_adc, valid_dwi_series)
        self.cleaning_details[study_name] = check_results
        return True, check_results
    
    def _save_valid_series(self, study_uid: str, t2_series: List, adc_series: List, dwi_series: List) -> None:
        study_output_dir = os.path.join(self.output_dir, study_uid)
        os.makedirs(study_output_dir, exist_ok=True)
        
        for i, (series_uid, files) in enumerate(t2_series):
            series_dir = os.path.join(study_output_dir, f"T2_TRA_{i+1}")
            os.makedirs(series_dir, exist_ok=True)
            for file in files:
                shutil.copy(file, os.path.join(series_dir, os.path.basename(file)))
        
        for i, (series_uid, files) in enumerate(adc_series):
            series_dir = os.path.join(study_output_dir, f"ADC_{i+1}")
            os.makedirs(series_dir, exist_ok=True)
            for file in files:
                shutil.copy(file, os.path.join(series_dir, os.path.basename(file)))
        
        for i, (series_uid, files) in enumerate(dwi_series):
            series_dir = os.path.join(study_output_dir, f"DWI_b{self.min_b_value}_{i+1}")
            os.makedirs(series_dir, exist_ok=True)
            for file in files:
                shutil.copy(file, os.path.join(series_dir, os.path.basename(file)))
    
    def process_directory(self) -> pd.DataFrame:
        print(f"开始处理目录: {self.input_dir}")
        study_dirs = [os.path.join(self.input_dir, d) for d in os.listdir(self.input_dir) 
                     if os.path.isdir(os.path.join(self.input_dir, d))]
        
        total_studies = len(study_dirs)
        valid_studies = 0
        
        for i, study_dir in enumerate(study_dirs, 1):
            study_name = os.path.basename(study_dir)
            print(f"\n处理Study {i}/{total_studies}: {study_name}")
            is_valid, _ = self.process_study(study_dir)
            if is_valid:
                valid_studies += 1
        
        print(f"\n处理完成!")
        print(f"总Studies: {total_studies}")
        print(f"有效Studies: {valid_studies}")
        print(f"无效Studies: {total_studies - valid_studies}")
        print(f"有效数据已保存至: {self.output_dir}")
        
        return self.get_report_dataframe()
    
    def get_report_dataframe(self) -> pd.DataFrame:
        """生成包含所有条件列的DataFrame"""
        data = []
        for study_name, check_results in self.cleaning_details.items():
            # 拼接失败原因（多个原因用换行分隔）
            failure_reasons = "\n".join(check_results["失败原因"]) if check_results["失败原因"] else ""
            # 构造一行数据（Study名称+所有检查条件）
            row_data = {"Study名称": study_name}
            row_data.update(check_results)  # 合并所有条件检查结果
            row_data["失败原因"] = failure_reasons  # 替换列表为字符串
            data.append(row_data)
        
        return pd.DataFrame(data)
    
    def save_report_to_excel(self, excel_path: str) -> None:
        report_df = self.get_report_dataframe()
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            report_df.to_excel(writer, sheet_name='详细检查报告', index=False)
            worksheet = writer.sheets['详细检查报告']
            
            # 调整列宽
            for i, col in enumerate(report_df.columns):
                max_length = max(report_df[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.column_dimensions[chr(65 + i)].width = max_length
            
            # 定义样式（True为绿色，False为红色）



inputpath0 = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train"
outputpath0 = "/kaggle/working/rsna-miccai-brain-tumor-radiogenomic-classification/train_cleaned"

for n,i in enumerate(os.listdir(inputpath0)):
    if n>=20:break
    inputpath1 = os.path.join(inputpath0,i)
    outputpath1 = os.path.join(outputpath0,i)
    excel = outputpath0+"//"+i+".xlsx"
    cleaner = DicomDataCleaner(inputpath1, outputpath1)
    report_df = cleaner.process_directory()
    cleaner.save_report_to_excel(excel)



import pandas as pd
cleanreportcase = """/kaggle/working/rsna-miccai-brain-tumor-radiogenomic-classification/train_cleaned/00688.xlsx"""
excelfile = pd.ExcelFile(cleanreportcase)
excelfile.parse(excelfile.sheet_names[0])




