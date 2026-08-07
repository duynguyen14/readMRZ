IF OBJECT_ID(N'dbo.readmrz_image_type_dataset_items', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.readmrz_image_type_dataset_items (
        Id BIGINT IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_readmrz_image_type_dataset_items PRIMARY KEY,

        SourceTable NVARCHAR(128) NOT NULL
            CONSTRAINT DF_readmrz_image_type_dataset_items_SourceTable
            DEFAULT (N'TransactionEVisa'),

        TransactionEVisaId BIGINT NOT NULL,
        TransactionGuid UNIQUEIDENTIFIER NULL,

        SourceField NVARCHAR(64) NOT NULL,
        SourceImageValue NVARCHAR(1000) NULL,

        Label NVARCHAR(32) NOT NULL,
        LabelId INT NOT NULL,

        RelativeImagePath NVARCHAR(1000) NOT NULL,
        ImageWidth INT NULL,
        ImageHeight INT NULL,
        FileSizeBytes BIGINT NULL,
        Sha256 NVARCHAR(64) NULL,

        Split NVARCHAR(16) NULL,
        Status NVARCHAR(32) NOT NULL
            CONSTRAINT DF_readmrz_image_type_dataset_items_Status
            DEFAULT (N'pending'),

        ErrorMessage NVARCHAR(2000) NULL,

        CreatedDate DATETIME2(0) NOT NULL
            CONSTRAINT DF_readmrz_image_type_dataset_items_CreatedDate
            DEFAULT SYSDATETIME(),

        UpdatedDate DATETIME2(0) NOT NULL
            CONSTRAINT DF_readmrz_image_type_dataset_items_UpdatedDate
            DEFAULT SYSDATETIME(),

        CONSTRAINT CK_readmrz_image_type_dataset_items_Label
            CHECK (Label IN (N'passport', N'face', N'EVISA_RESULT', N'VOA_RESULT')),

        CONSTRAINT CK_readmrz_image_type_dataset_items_LabelId
            CHECK (LabelId IN (0, 1, 2, 3)),

        CONSTRAINT CK_readmrz_image_type_dataset_items_SourceField
            CHECK (SourceField IN (N'FullPassportImage', N'FaceImage', N'FileEVisa', N'FilePath')),

        CONSTRAINT CK_readmrz_image_type_dataset_items_Status
            CHECK (Status IN (N'pending', N'copied', N'error', N'skipped')),

        CONSTRAINT CK_readmrz_image_type_dataset_items_Split
            CHECK (Split IS NULL OR Split IN (N'train', N'val', N'test'))
    );
END;
GO

IF EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = N'CK_readmrz_image_type_dataset_items_Label'
      AND parent_object_id = OBJECT_ID(N'dbo.readmrz_image_type_dataset_items')
)
BEGIN
    ALTER TABLE dbo.readmrz_image_type_dataset_items
    DROP CONSTRAINT CK_readmrz_image_type_dataset_items_Label;
END;
GO

ALTER TABLE dbo.readmrz_image_type_dataset_items
ADD CONSTRAINT CK_readmrz_image_type_dataset_items_Label
CHECK (Label IN (N'passport', N'face', N'EVISA_RESULT', N'VOA_RESULT'));
GO

IF EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = N'CK_readmrz_image_type_dataset_items_LabelId'
      AND parent_object_id = OBJECT_ID(N'dbo.readmrz_image_type_dataset_items')
)
BEGIN
    ALTER TABLE dbo.readmrz_image_type_dataset_items
    DROP CONSTRAINT CK_readmrz_image_type_dataset_items_LabelId;
END;
GO

ALTER TABLE dbo.readmrz_image_type_dataset_items
ADD CONSTRAINT CK_readmrz_image_type_dataset_items_LabelId
CHECK (LabelId IN (0, 1, 2, 3));
GO

IF EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = N'CK_readmrz_image_type_dataset_items_SourceField'
      AND parent_object_id = OBJECT_ID(N'dbo.readmrz_image_type_dataset_items')
)
BEGIN
    ALTER TABLE dbo.readmrz_image_type_dataset_items
    DROP CONSTRAINT CK_readmrz_image_type_dataset_items_SourceField;
END;
GO

ALTER TABLE dbo.readmrz_image_type_dataset_items
ADD CONSTRAINT CK_readmrz_image_type_dataset_items_SourceField
CHECK (SourceField IN (N'FullPassportImage', N'FaceImage', N'FileEVisa', N'FilePath'));
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'UX_readmrz_image_type_dataset_items_source_field'
      AND object_id = OBJECT_ID(N'dbo.readmrz_image_type_dataset_items')
)
BEGIN
    CREATE UNIQUE INDEX UX_readmrz_image_type_dataset_items_source_field
    ON dbo.readmrz_image_type_dataset_items (TransactionEVisaId, SourceField);
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_readmrz_image_type_dataset_items_Label_Status'
      AND object_id = OBJECT_ID(N'dbo.readmrz_image_type_dataset_items')
)
BEGIN
    CREATE INDEX IX_readmrz_image_type_dataset_items_Label_Status
    ON dbo.readmrz_image_type_dataset_items (Label, Status, Id);
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_readmrz_image_type_dataset_items_Split_Label'
      AND object_id = OBJECT_ID(N'dbo.readmrz_image_type_dataset_items')
)
BEGIN
    CREATE INDEX IX_readmrz_image_type_dataset_items_Split_Label
    ON dbo.readmrz_image_type_dataset_items (Split, Label, Id);
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_readmrz_image_type_dataset_items_Sha256'
      AND object_id = OBJECT_ID(N'dbo.readmrz_image_type_dataset_items')
)
BEGIN
    CREATE INDEX IX_readmrz_image_type_dataset_items_Sha256
    ON dbo.readmrz_image_type_dataset_items (Sha256)
    WHERE Sha256 IS NOT NULL;
END;
GO
