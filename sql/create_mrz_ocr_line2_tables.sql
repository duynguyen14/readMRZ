IF OBJECT_ID(N'dbo.readmrz_label_items', N'U') IS NULL
BEGIN
    THROW 50001, 'Missing dbo.readmrz_label_items. Run tools/create_mrz_label_review_tables.py first.', 1;
END;

IF COL_LENGTH(N'dbo.readmrz_label_items', N'mrz_line2_extract_status') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_label_items
    ADD mrz_line2_extract_status VARCHAR(32) NULL,
        mrz_line2_extract_count INT NULL,
        mrz_line2_extract_error NVARCHAR(2000) NULL,
        mrz_line2_extracted_at DATETIME2 NULL;
END;

IF OBJECT_ID(N'dbo.readmrz_ocr_line_items2', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.readmrz_ocr_line_items2 (
        id BIGINT IDENTITY(1,1) NOT NULL PRIMARY KEY,
        label_item_id BIGINT NOT NULL,
        source_key NVARCHAR(700) NOT NULL,
        split VARCHAR(20) NULL,
        line_index INT NOT NULL,

        mrz_crop_file_name NVARCHAR(700) NULL,
        line_image_file_name NVARCHAR(700) NOT NULL,
        line_image_width INT NULL,
        line_image_height INT NULL,

        mrz_bbox_x1 FLOAT NULL,
        mrz_bbox_y1 FLOAT NULL,
        mrz_bbox_x2 FLOAT NULL,
        mrz_bbox_y2 FLOAT NULL,
        line_bbox_x1 FLOAT NULL,
        line_bbox_y1 FLOAT NULL,
        line_bbox_x2 FLOAT NULL,
        line_bbox_y2 FLOAT NULL,

        deskew_angle FLOAT NULL,
        projection_score FLOAT NULL,
        split_method VARCHAR(64) NOT NULL CONSTRAINT DF_readmrz_ocr_line_items2_split_method DEFAULT ('opencv_projection'),

        ocr_text NVARCHAR(160) NULL,
        normalized_text NVARCHAR(160) NULL,
        final_text NVARCHAR(160) NULL,
        ocr_score FLOAT NULL,
        mrz_likeness FLOAT NULL,

        review_status VARCHAR(32) NOT NULL CONSTRAINT DF_readmrz_ocr_line_items2_review_status DEFAULT ('pending'),
        error_message NVARCHAR(2000) NULL,

        created_at DATETIME2 NOT NULL CONSTRAINT DF_readmrz_ocr_line_items2_created_at DEFAULT SYSUTCDATETIME(),
        reviewed_at DATETIME2 NULL,
        updated_at DATETIME2 NOT NULL CONSTRAINT DF_readmrz_ocr_line_items2_updated_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT FK_readmrz_ocr_line_items2_label_item
            FOREIGN KEY (label_item_id) REFERENCES dbo.readmrz_label_items(id),
        CONSTRAINT UQ_readmrz_ocr_line_items2_label_line UNIQUE (label_item_id, line_index),
        CONSTRAINT CK_readmrz_ocr_line_items2_review_status CHECK (review_status IN ('pending', 'approved', 'rejected'))
    );
END;

IF COL_LENGTH(N'dbo.readmrz_ocr_line_items2', N'mrz_crop_file_name') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_ocr_line_items2 ADD mrz_crop_file_name NVARCHAR(700) NULL;
END;

IF COL_LENGTH(N'dbo.readmrz_ocr_line_items2', N'line_image_width') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_ocr_line_items2 ADD line_image_width INT NULL;
END;

IF COL_LENGTH(N'dbo.readmrz_ocr_line_items2', N'line_image_height') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_ocr_line_items2 ADD line_image_height INT NULL;
END;

IF COL_LENGTH(N'dbo.readmrz_ocr_line_items2', N'mrz_bbox_x1') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_ocr_line_items2 ADD mrz_bbox_x1 FLOAT NULL;
END;

IF COL_LENGTH(N'dbo.readmrz_ocr_line_items2', N'mrz_bbox_y1') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_ocr_line_items2 ADD mrz_bbox_y1 FLOAT NULL;
END;

IF COL_LENGTH(N'dbo.readmrz_ocr_line_items2', N'mrz_bbox_x2') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_ocr_line_items2 ADD mrz_bbox_x2 FLOAT NULL;
END;

IF COL_LENGTH(N'dbo.readmrz_ocr_line_items2', N'mrz_bbox_y2') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_ocr_line_items2 ADD mrz_bbox_y2 FLOAT NULL;
END;

IF COL_LENGTH(N'dbo.readmrz_ocr_line_items2', N'line_bbox_x1') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_ocr_line_items2 ADD line_bbox_x1 FLOAT NULL;
END;

IF COL_LENGTH(N'dbo.readmrz_ocr_line_items2', N'line_bbox_y1') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_ocr_line_items2 ADD line_bbox_y1 FLOAT NULL;
END;

IF COL_LENGTH(N'dbo.readmrz_ocr_line_items2', N'line_bbox_x2') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_ocr_line_items2 ADD line_bbox_x2 FLOAT NULL;
END;

IF COL_LENGTH(N'dbo.readmrz_ocr_line_items2', N'line_bbox_y2') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_ocr_line_items2 ADD line_bbox_y2 FLOAT NULL;
END;

IF COL_LENGTH(N'dbo.readmrz_ocr_line_items2', N'doc_orientation_angle') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_ocr_line_items2
    ADD doc_orientation_angle INT NOT NULL CONSTRAINT DF_readmrz_ocr_line_items2_doc_orientation_angle DEFAULT (0);
END;

IF COL_LENGTH(N'dbo.readmrz_ocr_line_items2', N'deskew_angle') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_ocr_line_items2 ADD deskew_angle FLOAT NULL;
END;

IF COL_LENGTH(N'dbo.readmrz_ocr_line_items2', N'projection_score') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_ocr_line_items2 ADD projection_score FLOAT NULL;
END;

IF COL_LENGTH(N'dbo.readmrz_ocr_line_items2', N'split_method') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_ocr_line_items2 ADD split_method VARCHAR(64) NULL;
END;

IF COL_LENGTH(N'dbo.readmrz_ocr_line_items2', N'ocr_text') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_ocr_line_items2 ADD ocr_text NVARCHAR(160) NULL;
END;

IF COL_LENGTH(N'dbo.readmrz_ocr_line_items2', N'normalized_text') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_ocr_line_items2 ADD normalized_text NVARCHAR(160) NULL;
END;

IF COL_LENGTH(N'dbo.readmrz_ocr_line_items2', N'final_text') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_ocr_line_items2 ADD final_text NVARCHAR(160) NULL;
END;

IF COL_LENGTH(N'dbo.readmrz_ocr_line_items2', N'ocr_score') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_ocr_line_items2 ADD ocr_score FLOAT NULL;
END;

IF COL_LENGTH(N'dbo.readmrz_ocr_line_items2', N'mrz_likeness') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_ocr_line_items2 ADD mrz_likeness FLOAT NULL;
END;

IF COL_LENGTH(N'dbo.readmrz_ocr_line_items2', N'error_message') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_ocr_line_items2 ADD error_message NVARCHAR(2000) NULL;
END;

IF COL_LENGTH(N'dbo.readmrz_ocr_line_items2', N'created_at') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_ocr_line_items2 ADD created_at DATETIME2 NULL;
END;

IF COL_LENGTH(N'dbo.readmrz_ocr_line_items2', N'reviewed_at') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_ocr_line_items2 ADD reviewed_at DATETIME2 NULL;
END;

IF COL_LENGTH(N'dbo.readmrz_ocr_line_items2', N'updated_at') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_ocr_line_items2 ADD updated_at DATETIME2 NULL;
END;

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_readmrz_ocr_line_items2_review_queue'
      AND object_id = OBJECT_ID(N'dbo.readmrz_ocr_line_items2')
)
BEGIN
    CREATE INDEX IX_readmrz_ocr_line_items2_review_queue
    ON dbo.readmrz_ocr_line_items2 (review_status, id)
    INCLUDE (source_key, split, line_index, line_image_file_name);
END;

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_readmrz_ocr_line_items2_source_key'
      AND object_id = OBJECT_ID(N'dbo.readmrz_ocr_line_items2')
)
BEGIN
    CREATE INDEX IX_readmrz_ocr_line_items2_source_key
    ON dbo.readmrz_ocr_line_items2 (source_key, line_index);
END;

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_readmrz_label_items_line2_extract'
      AND object_id = OBJECT_ID(N'dbo.readmrz_label_items')
)
BEGIN
    CREATE INDEX IX_readmrz_label_items_line2_extract
    ON dbo.readmrz_label_items (review_status, mrz_line2_extract_status, id)
    INCLUDE (source_key, split, image_file_name);
END;
